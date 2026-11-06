"""Combined: layer-wise back-to-front projection + multi-restart L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_H_cache = {}
_e0_cache = {}
_cz_mask_cache = {}
_idx_cache = {}


def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    # Z terms
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
    # X terms
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    _H_cache[key] = H
    return H


def _exact_ground_state(n, coupling, field):
    key = (n, coupling, field)
    if key in _e0_cache:
        return _e0_cache[key]
    H = _build_H(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    e0 = eigvals[0]
    psi0 = eigvecs[:, 0]
    _e0_cache[key] = (e0, psi0)
    return e0, psi0


def _get_cz_masks(n):
    key = n
    if key in _cz_mask_cache:
        return _cz_mask_cache[key]
    dim = 1 << n
    arange = np.arange(dim)
    masks = []
    for i in range(n - 1):
        mask1 = 1 << i
        mask2 = 1 << (i + 1)
        both_set = ((arange & mask1) != 0) & ((arange & mask2) != 0)
        masks.append(both_set)
    _cz_mask_cache[key] = masks
    return masks


def _get_idx_pairs(n):
    key = n
    if key in _idx_cache:
        return _idx_cache[key]
    dim = 1 << n
    idx_0 = {}
    idx_1 = {}
    for q in range(n):
        mask = 1 << q
        idx_0[q] = np.array([i for i in range(dim) if (i & mask) == 0])
        idx_1[q] = idx_0[q] + mask
    _idx_cache[key] = (idx_0, idx_1)
    return (idx_0, idx_1)


def _circuit_state_fast(n, thetas, layers):
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j

    idx_0, idx_1 = _get_idx_pairs(n)
    cz_masks = _get_cz_masks(n)

    for layer in range(layers):
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)

            i0 = idx_0[q]
            i1 = idx_1[q]

            vals_0 = state[i0]
            vals_1 = state[i1]

            state[i0] = c * vals_0 - s * vals_1
            state[i1] = s * vals_0 + c * vals_1

        if layer < layers - 1:
            for m in cz_masks:
                state[m] *= -1.0

    return state


def _circuit_state_forward(n, thetas, layers, start_layer=0):
    """
    Apply the circuit from start_layer to layers-1.
    If start_layer == 0, start from |0>^n.
    Returns the final state.
    """
    dim = 1 << n
    if start_layer == 0:
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0 + 0j
    else:
        raise ValueError("start_layer must be 0 for forward pass from |0>")

    idx_0, idx_1 = _get_idx_pairs(n)
    cz_masks = _get_cz_masks(n)

    for layer in range(start_layer, layers):
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)

            i0 = idx_0[q]
            i1 = idx_1[q]

            vals_0 = state[i0]
            vals_1 = state[i1]

            state[i0] = c * vals_0 - s * vals_1
            state[i1] = s * vals_0 + c * vals_1

        if layer < layers - 1:
            for m in cz_masks:
                state[m] *= -1.0

    return state


def _apply_inverse_circuit(n, thetas, layers, state):
    """
    Apply the inverse of the circuit to a state.
    The circuit is: Ry layers with CZ between.
    Inverse: reverse order, inverse CZ (same as CZ), inverse Ry.
    We apply from last layer to first, and within each layer reverse qubit order for Ry.
    Actually, the circuit applies Ry to all qubits in a layer (in any order since they're on different qubits),
    then CZ. The inverse applies CZ first (same), then inverse Ry.
    Since CZ is self-inverse and diagonal, and Ry gates on different qubits commute,
    the inverse of layer l's Ry is just Ry(-theta) on the same qubits.
    
    But wait: the circuit structure is:
    Layer 0: Ry on all qubits, then CZ
    Layer 1: Ry on all qubits, then CZ
    Layer 2: Ry on all qubits (no CZ after)
    
    So the full circuit is: Ry(2) CZ Ry(1) CZ Ry(0)
    Inverse is: Ry(0)^{-1} CZ^{-1} Ry(1)^{-1} CZ^{-1} Ry(2)^{-1}
    = Ry(-theta_0) CZ Ry(-theta_1) CZ Ry(-theta_2)
    
    So to apply inverse, we go from last layer to first:
    For layer = layers-1 down to 0:
        Apply Ry(-theta[layer, q]) for all q
        If layer > 0, apply CZ (since CZ is self-inverse)
    
    Wait, let me re-think. The forward circuit is:
    state -> Ry(layer0) -> CZ -> Ry(layer1) -> CZ -> Ry(layer2)
    
    So U = Ry2 * CZ * Ry1 * CZ * Ry0
    
    U^{-1} = Ry0^{-1} * CZ^{-1} * Ry1^{-1} * CZ^{-1} * Ry2^{-1}
    
    So to apply U^{-1} to a state:
    First apply Ry2^{-1} (i.e., Ry(-theta_2))
    Then apply CZ
    Then apply Ry1^{-1} (i.e., Ry(-theta_1))
    Then apply CZ
    Then apply Ry0^{-1} (i.e., Ry(-theta_0))
    
    So we iterate from layer = layers-1 down to 0:
        Apply Ry(-theta[layer, q]) for all q
        If layer > 0, apply CZ
    """
    dim = 1 << n
    idx_0, idx_1 = _get_idx_pairs(n)
    cz_masks = _get_cz_masks(n)

    for layer in range(layers - 1, -1, -1):
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(-theta / 2.0)
            s = np.sin(-theta / 2.0)

            i0 = idx_0[q]
            i1 = idx_1[q]

            vals_0 = state[i0]
            vals_1 = state[i1]

            state[i0] = c * vals_0 - s * vals_1
            state[i1] = s * vals_0 + c * vals_1

        if layer > 0:
            for m in cz_masks:
                state[m] *= -1.0

    return state


def _circuit_energy(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state_fast(n, thetas, layers)
    return np.real(np.conj(psi) @ H @ psi)


def _objective(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n


def _layer_wise_projection(n, coupling, field, layers, psi_target, deadline):
    """
    Back-to-front layer-wise projection.
    For each layer l from last to first:
      1. Fix all other layers' angles to their current best.
      2. Compute the "effective target" for this layer by applying the inverse
         of the subsequent layers to psi_target.
      3. Optimize this layer's Ry angles to minimize distance to the effective target.
    """
    dim = 1 << n
    idx_0, idx_1 = _get_idx_pairs(n)
    cz_masks = _get_cz_masks(n)

    # Initialize with heuristic angles
    thetas = np.zeros((layers, n))
    ratio = field / coupling if coupling != 0 else field
    theta_est = (np.pi / 2) * (1 - np.exp(-ratio))
    thetas[:] = theta_est

    # Iterate back-to-front a few times
    for iteration in range(3):
        if time.time() > deadline:
            break

        # For each layer from last to first
        for l in range(layers - 1, -1, -1):
            if time.time() > deadline:
                break

            # Compute the effective target for layer l
            # The state after layer l should be such that when we apply
            # layers l+1 to layers-1, we get psi_target.
            # So: U_{l+1..} * |psi_after_l> = psi_target
            # |psi_after_l> = U_{l+1..}^{-1} * psi_target
            
            # U_{l+1..} is the circuit from layer l+1 to layers-1
            # We need to apply its inverse to psi_target.
            
            # The sub-circuit from layer l+1 to layers-1 is:
            # If l+1 == layers-1: just Ry(l+1)
            # If l+1 < layers-1: Ry(l+1) CZ Ry(l+2) CZ ... CZ Ry(layers-1)
            
            # Let's build the thetas for the sub-circuit
            sub_thetas = thetas[l+1:, :]
            sub_layers = layers - (l + 1)
            
            if sub_layers == 0:
                # No subsequent layers, target is psi_target directly
                effective_target = psi_target.copy()
            else:
                # Apply inverse of sub-circuit to psi_target
                effective_target = _apply_inverse_circuit(n, sub_thetas, sub_layers, psi_target.copy())
            
            # Now we want to find theta_l such that:
            # Ry(theta_l) * |psi_before_l> ≈ effective_target
            # where |psi_before_l> is the state before layer l.
            
            # For l=0, |psi_before_0> = |0>^n
            # For l>0, |psi_before_l> is the state after CZ following layer l-1.
            # But we don't know it yet. Let's approximate:
            # We can compute |psi_before_l> by applying the forward circuit up to layer l-1.
            
            if l == 0:
                psi_before = np.zeros(dim, dtype=np.complex128)
                psi_before[0] = 1.0 + 0j
            else:
                # Compute state after layer l-1 (including CZ if l-1 < layers-1)
                # This is the state after applying layers 0 to l-1
                psi_before = _circuit_state_forward(n, thetas[:l, :], l)
                # Wait, _circuit_state_forward applies from layer 0 to layers-1.
                # We need to apply only up to layer l-1.
                # Let me write a helper.
                
                # Actually, let's just compute it directly
                psi_before = np.zeros(dim, dtype=np.complex128)
                psi_before[0] = 1.0 + 0j
                for layer in range(l):
                    for q in range(n):
                        theta = thetas[layer, q]
                        c = np.cos(theta / 2.0)
                        s = np.sin(theta / 2.0)
                        i0 = idx_0[q]
                        i1 = idx_1[q]
                        vals_0 = psi_before[i0]
                        vals_1 = psi_before[i1]
                        psi_before[i0] = c * vals_0 - s * vals_1
                        psi_before[i1] = s * vals_0 + c * vals_1
                    if layer < layers - 1:
                        for m in cz_masks:
                            psi_before[m] *= -1.0
            
            # Now optimize theta_l to minimize ||Ry(theta_l) psi_before - effective_target||^2
            # This is equivalent to maximizing Re[<effective_target| Ry(theta_l) |psi_before>]
            
            def _layer_obj(theta_flat):
                theta_arr = theta_flat.reshape((n,))
                # Apply Ry(theta_arr) to psi_before
                psi_after = psi_before.copy()
                for q in range(n):
                    theta = theta_arr[q]
                    c = np.cos(theta / 2.0)
                    s = np.sin(theta / 2.0)
                    i0 = idx_0[q]
                    i1 = idx_1[q]
                    vals_0 = psi_after[i0]
                    vals_1 = psi_after[i1]
                    psi_after[i0] = c * vals_0 - s * vals_1
                    psi_after[i1] = s * vals_0 + c * vals_1
                # Distance to effective_target
                diff = psi_after - effective_target
                return np.real(np.vdot(diff, diff))
            
            # Multi-restart for this layer
            best_theta = thetas[l, :].copy()
            best_val = _layer_obj(best_theta)
            
            inits = [best_theta.copy()]
            inits.append(np.zeros(n))
            inits.append(np.full(n, np.pi / 2))
            inits.append(np.full(n, np.pi))
            for i in range(4):
                np.random.seed(500 + l * 10 + i)
                inits.append(np.random.uniform(-np.pi, np.pi, size=(n,)))
            
            for init in inits:
                if time.time() > deadline:
                    break
                try:
                    res = minimize(_layer_obj, init, method='L-BFGS-B',
                                  options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                    if res.fun < best_val:
                        best_val = res.fun
                        best_theta = res.x.copy()
                except Exception:
                    pass
            
            thetas[l, :] = best_theta

    return thetas


def _heuristic_init(n, coupling, field, layers):
    """
    Heuristic initialization based on ground state marginals.
    """
    H = _build_H(n, coupling, field)
    dim = 1 << n
    eigvals, eigvecs = np.linalg.eigh(H)
    psi0 = eigvecs[:, 0]
    
    # Estimate single-qubit excitation probabilities
    # For each qubit q, compute <psi0| X_q |psi0> or similar
    # Actually, for a product state approximation, we want to find theta such that
    # cos^2(theta/2) ≈ P(0) and sin^2(theta/2) ≈ P(1)
    
    thetas = np.zeros((layers, n))
    for q in range(n):
        # Compute probability of qubit q being 1
        mask = 1 << q
        prob_1 = 0.0
        for state in range(dim):
            if (state & mask) != 0:
                prob_1 += np.abs(psi0[state]) ** 2
        # For Ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
        # So sin^2(theta/2) = prob_1
        # theta = 2 * arcsin(sqrt(prob_1))
        if prob_1 > 0 and prob_1 < 1:
            theta = 2 * np.arcsin(np.sqrt(prob_1))
        else:
            theta = 0.0
        thetas[:, q] = theta
    
    return thetas


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0

    e0, psi0 = _exact_ground_state(n, coupling, field)

    best_x = np.zeros(layers * n)
    best_val = _objective(best_x, n, coupling, field, layers, e0)

    # Layer-wise projection initialization
    try:
        if time.time() < deadline - 30.0:
            thetas_proj = _layer_wise_projection(n, coupling, field, layers, psi0, deadline - 30.0)
            val_proj = _objective(thetas_proj.flatten(), n, coupling, field, layers, e0)
            if val_proj < best_val:
                best_val = val_proj
                best_x = thetas_proj.flatten().copy()
    except Exception:
        pass

    # Heuristic initialization
    try:
        if time.time() < deadline - 30.0:
            thetas_heur = _heuristic_init(n, coupling, field, layers)
            val_heur = _objective(thetas_heur.flatten(), n, coupling, field, layers, e0)
            if val_heur < best_val:
                best_val = val_heur
                best_x = thetas_heur.flatten().copy()
    except Exception:
        pass

    # Build diverse initializations
    inits = []
    inits.append(np.zeros(layers * n))
    inits.append(np.full(layers * n, np.pi / 2))
    inits.append(np.full(layers * n, np.pi))
    inits.append(np.full(layers * n, -np.pi / 2))

    ratio = field / coupling if coupling != 0 else field
    theta_est = (np.pi / 2) * (1 - np.exp(-ratio))
    inits.append(np.full(layers * n, theta_est))

    inits.append(np.full(layers * n, 0.5))
    inits.append(np.full(layers * n, 1.0))
    inits.append(np.full(layers * n, 1.5))

    t1 = np.full(layers * n, 0.3)
    t1[n:2*n] = 0.8
    t1[2*n:3*n] = 0.5
    inits.append(t1)

    t2 = np.full(layers * n, 0.7)
    t2[n:2*n] = 0.4
    t2[2*n:3*n] = 0.9
    inits.append(t2)

    for i in range(12):
        np.random.seed(100 + i)
        inits.append(np.random.uniform(-np.pi, np.pi, size=(layers * n)))

    for i in range(8):
        np.random.seed(200 + i)
        inits.append(np.random.uniform(0, np.pi, size=(layers * n)))

    max_restarts = 30 if n <= 4 else 24
    num_restarts = min(len(inits), max_restarts)

    for i in range(num_restarts):
        if time.time() > deadline:
            break

        x_init = inits[i].copy()

        try:
            res = minimize(lambda p: _objective(p, n, coupling, field, layers, e0),
                          x_init, method='L-BFGS-B',
                          options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Polish phase
    remaining = deadline - time.time()
    if remaining > 5.0:
        try:
            res = minimize(lambda p: _objective(p, n, coupling, field, layers, e0),
                          best_x, method='L-BFGS-B',
                          options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Second polish from a slightly perturbed best
    remaining = deadline - time.time()
    if remaining > 5.0:
        try:
            np.random.seed(42)
            perturbed = best_x + np.random.normal(0, 0.1, size=(layers * n))
            res = minimize(lambda p: _objective(p, n, coupling, field, layers, e0),
                          perturbed, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    return best_x.reshape((layers, n))
# EVOLVE-BLOCK-END
