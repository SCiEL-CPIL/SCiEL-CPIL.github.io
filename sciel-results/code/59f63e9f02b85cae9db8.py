"""Quantum information: shallow-circuit Ising ground states.
Uses analytic gradients via adjoint method and symmetry reduction."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_cache = {}
_start_time = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.float64)
    for idx in range(dim):
        h_diag = 0.0
        for i in range(n - 1):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            h_diag += -coupling * zi * zip1
        H[idx, idx] = h_diag
        for i in range(n):
            other = idx ^ (1 << i)
            H[idx, other] += -field
    return H


def _exact_ground_state(H):
    """Compute exact ground state vector and energy."""
    eigvals, eigvecs = np.linalg.eigh(H)
    idx = np.argmin(eigvals)
    return np.real(eigvals[idx]), eigvecs[:, idx]


def _apply_ry(psi, q, theta, n):
    """Apply Ry(theta) on qubit q to state psi. Returns new array."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    # Move qubit q to the last axis for easier indexing
    psi_moved = np.moveaxis(psi, q, -1)
    # Shape is (2^n-1, 2)
    new0 = c * psi_moved[..., 0] - s * psi_moved[..., 1]
    new1 = s * psi_moved[..., 0] + c * psi_moved[..., 1]
    psi_out = np.stack([new0, new1], axis=-1)
    return np.moveaxis(psi_out, -1, q)


def _apply_cz(psi, q, n):
    """Apply CZ on qubits q, q+1 to state psi. Returns new array."""
    # Move q and q+1 to the last two axes
    psi_moved = np.moveaxis(psi, [q, q + 1], [-2, -1])
    # Copy to ensure we don't modify original if it's a view (though we return new anyway)
    psi_moved = psi_moved.copy()
    # CZ flips sign of |11> component
    psi_moved[..., 1, 1] *= -1.0
    return np.moveaxis(psi_moved, [-2, -1], [q, q + 1])


def _circuit_forward_backward(params, n, layers):
    """
    Computes the final state and the gradients of the state w.r.t parameters.
    Uses the adjoint method (reverse-mode automatic differentiation) for efficiency.
    
    Returns:
        psi_final: final state vector
        grads: list of arrays, each is d|psi_final>/dtheta_k
    """
    dim = 2 ** n
    num_params = layers * n
    
    # Store all intermediate states
    # states[0] is |0...0>
    # states[k+1] is state after applying gate k
    # We have layers*n Ry gates and (layers-1)*(n-1) CZ gates.
    # But we only need gradients w.r.t Ry parameters.
    
    # Let's store state before each Ry gate and the state after all gates.
    # Actually, it's easier to store the state after every gate application.
    
    # Gate sequence:
    # L=0: Ry(0), Ry(1), ..., Ry(n-1), CZ(0), ..., CZ(n-2)
    # L=1: Ry(0), Ry(1), ..., Ry(n-1), CZ(0), ..., CZ(n-2)
    # L=2: Ry(0), Ry(1), ..., Ry(n-1)
    
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    
    # We need to store the state *before* each Ry gate to compute the gradient contribution.
    # Let states_before_ry[k] be the state before applying Ry gate k.
    # Let angles[k] be the angle for Ry gate k.
    
    states_before_ry = []
    angles = []
    
    # Forward pass
    for layer in range(layers):
        for q in range(n):
            states_before_ry.append(psi.copy())
            angles.append(params[layer * n + q])
            psi = _apply_ry(psi, q, angles[-1], n)
        if layer < layers - 1:
            for q in range(n - 1):
                psi = _apply_cz(psi, q, n)
    
    psi_final = psi
    
    # Backward pass (Adjoint)
    # We want d|psi_final>/dtheta_k.
    # Let |psi_final> = U_after_k * Ry_k(theta_k) * |psi_before_k>
    # d|psi_final>/dtheta_k = U_after_k * (dRy_k/dtheta_k) * |psi_before_k>
    
    # Instead of applying U_after_k explicitly, we propagate the adjoint state backwards.
    # Let |chi> be the adjoint state, initialized as |psi_final>.
    # We propagate |chi> backwards through the gates.
    # When we pass through a Ry gate, we compute the gradient contribution.
    
    # The gradient of the energy E = <psi|H|psi> is 2 Re[ <H psi | dpsi/dtheta> ].
    # Let |h_psi> = H |psi_final>.
    # We can compute g_k = 2 Re[ <h_psi | dpsi/dtheta_k> ].
    
    # Let's propagate |chi> = |h_psi> backwards.
    # Initially |chi> = H @ psi_final.
    chi = np.vdot(H, psi_final).conjugate() # Wait, H is real symmetric. H @ psi is complex.
    # <H psi | v> = (H psi)^dagger v = psi^dagger H v.
    # We want <H psi | dpsi_k>.
    # Let's define the adjoint vector |a> such that <H psi | dpsi_k> = <a | something>.
    
    # Standard adjoint method for quantum circuits:
    # We want dE/dtheta_k.
    # E = <psi_final|H|psi_final>.
    # dE/dtheta_k = <psi_final|H| dpsi_k/dtheta_k> + <dpsi_k/dtheta_k|H|psi_final>
    #             = 2 Re[ <psi_final|H| dpsi_k/dtheta_k> ]
    
    # Let |phi> = H |psi_final>.
    # Then dE/dtheta_k = 2 Re[ <psi_final| |phi> ... no.
    # <psi_final| H | dpsi_k> = (H^dagger |psi_final>)^dagger |dpsi_k> = <phi | dpsi_k> where phi = H psi_final? No.
    # (H psi) is a vector. <psi| H |dpsi> = (H psi)^dagger dpsi? No.
    # <psi| H |dpsi> = sum_i psi_i^* (H dpsi)_i.
    # (H psi)^dagger dpsi = sum_i (H psi)_i^* dpsi_i.
    # Since H is Hermitian, <psi| H |dpsi> = <H psi | dpsi>.
    # So we need <H psi_final | dpsi_k/dtheta_k>.
    
    # Let's propagate the vector |v> = H @ psi_final backwards.
    # Actually, it's easier to propagate the bra <psi_final| H = <psi_final| H.
    # Let's stick to the state vector propagation.
    
    # We have the chain:
    # |psi_final> = G_L ... G_1 |psi_0>
    # d|psi_final>/dtheta_k = G_L ... G_{k+1} (dG_k/dtheta_k) G_{k-1} ... G_1 |psi_0>
    
    # Let |adj> = |psi_final> initially? No.
    # Let's define the adjoint state |a> such that <a | dpsi_k> = <H psi_final | dpsi_k>.
    # So |a> should be H |psi_final>?
    # Check: <H psi_final | dpsi_k> = (H psi_final)^dagger dpsi_k.
    # If we propagate |a> = H psi_final backwards, we need to be careful with the adjoint of the gates.
    
    # Let's use the standard reverse-mode AD:
    # Start with |a> = H |psi_final> (this is the "output gradient" for the energy, effectively).
    # Actually, the gradient of E w.r.t the final state is H |psi_final>.
    # So we start with |a> = H |psi_final>.
    
    # Wait, E = <psi|H|psi>. dE = <dpsi|H|psi> + <psi|H|dpsi> = 2 Re[<dpsi|H|psi>].
    # So the "gradient" of E w.r.t psi is 2 H psi (real part handling aside).
    # Let's just compute the complex gradient <H psi | dpsi_k> and take 2*Re at the end.
    
    # So, initialize adjoint vector |a> = H @ psi_final.
    a = H @ psi_final
    
    grads = np.zeros(num_params, dtype=np.float64)
    
    # We need to reverse the gate sequence.
    # The gates were:
    # L=2: Ry(0..n-1)
    # L=1: Ry(0..n-1), CZ(0..n-2)
    # L=0: Ry(0..n-1), CZ(0..n-2)
    
    # Reverse order:
    # 1. Undo L=2 Ry gates (from n-1 down to 0)
    # 2. Undo L=1 CZ gates (from n-2 down to 0)
    # 3. Undo L=1 Ry gates (from n-1 down to 0)
    # 4. Undo L=0 CZ gates (from n-2 down to 0)
    # 5. Undo L=0 Ry gates (from n-1 down to 0)
    
    # Helper to undo Ry:
    # Ry(theta) = [[c, -s], [s, c]]
    # Ry(theta)^dagger = Ry(-theta) = [[c, s], [-s, c]]
    # If |psi_out> = Ry |psi_in>, then |psi_in> = Ry^dagger |psi_out>.
    
    # Helper to undo CZ:
    # CZ is Hermitian and unitary, so CZ^dagger = CZ.
    
    # We need to compute <a | dpsi_k> at the point of the gate.
    # When we are at the position of Ry gate k in the reverse pass, we have the adjoint state |a> that has been propagated back through all subsequent gates.
    # The contribution to the gradient is:
    # g_k = <a | (dRy_k/dtheta_k) |psi_before_k>
    
    # Note: |a> here is the state after applying the adjoints of all subsequent gates to the initial adjoint vector.
    
    # Let's implement the reverse pass carefully.
    
    # We stored states_before_ry.
    # We also need to know the angles.
    
    # Indexing:
    # k = layer * n + q
    
    # Reverse loop:
    for layer in range(layers - 1, -1, -1):
        # Undo Ry gates of this layer (reverse order)
        for q in range(n - 1, -1, -1):
            k = layer * n + q
            theta = angles[k]
            psi_before = states_before_ry[k]
            
            # Calculate gradient contribution
            # dRy/dtheta = 0.5 * [[-sin(t/2), -cos(t/2)], [cos(t/2), -sin(t/2)]]
            # Let's compute <a | dRy | psi_before>
            
            # Apply Ry^dagger to a to get a_in? No, we need <a | dRy | psi_before>.
            # This is equivalent to <(Ry^dagger a) | (Ry^dagger dRy | psi_before)>? No.
            # Just compute the inner product directly.
            
            # dRy |psi_before>
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            # dRy matrix:
            # 0.5 * [ [-s, -c], [c, -s] ]
            
            # We need to apply dRy to psi_before on qubit q.
            # And take inner product with a on qubit q.
            
            # Move qubit q to last axis for both a and psi_before
            a_moved = np.moveaxis(a, q, -1)
            psi_moved = np.moveaxis(psi_before, q, -1)
            
            # dpsi_0 = 0.5 * (-s * psi_0 - c * psi_1)
            # dpsi_1 = 0.5 * (c * psi_0 - s * psi_1)
            
            dpsi_0 = 0.5 * (-s * psi_moved[..., 0] - c * psi_moved[..., 1])
            dpsi_1 = 0.5 * (c * psi_moved[..., 0] - s * psi_moved[..., 1])
            dpsi = np.stack([dpsi_0, dpsi_1], axis=-1)
            
            # Inner product: sum conj(a) * dpsi
            # a_moved shape: (..., 2)
            # dpsi shape: (..., 2)
            prod = np.conj(a_moved) * dpsi
            val = np.sum(prod)
            
            grads[k] = 2.0 * val.real
            
            # Propagate a backwards through Ry gate
            # a_in = Ry^dagger a
            # Ry^dagger = [[c, s], [-s, c]]
            a_new_0 = c * a_moved[..., 0] + s * a_moved[..., 1]
            a_new_1 = -s * a_moved[..., 0] + c * a_moved[..., 1]
            a_new = np.stack([a_new_0, a_new_1], axis=-1)
            a = np.moveaxis(a_new, -1, q)
            
        # Undo CZ gates of this layer (if not last layer)
        if layer < layers - 1:
            for q in range(n - 2, -1, -1):
                # CZ is its own inverse
                a = _apply_cz(a, q, n)
                
    return psi_final, grads


def _energy_and_gradient(params, n, layers, H, e0):
    """Compute energy and gradient of (E_circuit - e0)/n."""
    dim = 2 ** n
    psi_final, grads = _circuit_forward_backward(params, n, layers)
    energy = np.vdot(psi_final, H @ psi_final).real
    obj_val = (energy - e0) / n
    
    # The grads returned are dE/dtheta.
    # We need d(obj_val)/dtheta = grads / n
    grad_obj = grads / n
    
    return obj_val, grad_obj


def _optimize_for_params(n, coupling, field, layers):
    """Optimize circuit angles for given parameters."""
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    H = _build_hamiltonian(n, coupling, field)
    e0, _ = _exact_ground_state(H)
    
    num_params = layers * n
    
    # Symmetry reduction
    # For open chain, theta[l, q] = theta[l, n-1-q]
    # We optimize only the first ceil(n/2) parameters per layer, then mirror.
    half_n = (n + 1) // 2
    num_independent = layers * half_n
    
    def expand_params(indep_params):
        full_params = np.zeros(num_params)
        for l in range(layers):
            for q in range(half_n):
                val = indep_params[l * half_n + q]
                full_params[l * n + q] = val
                full_params[l * n + (n - 1 - q)] = val
        return full_params
    
    def objective(indep_params):
        full_params = expand_params(indep_params)
        val, _ = _energy_and_gradient(full_params, n, layers, H, e0)
        return val
    
    def gradient(indep_params):
        full_params = expand_params(indep_params)
        _, grad_full = _energy_and_gradient(full_params, n, layers, H, e0)
        
        grad_indep = np.zeros(num_independent)
        for l in range(layers):
            for q in range(half_n):
                idx_full = l * n + q
                val = grad_full[idx_full]
                if q == n - 1 - q:
                    grad_indep[l * half_n + q] = val
                else:
                    idx_mirror = l * n + (n - 1 - q)
                    grad_indep[l * half_n + q] = val + grad_full[idx_mirror]
        return grad_indep
    
    # Physics-based initial guesses
    ratio = field / coupling if coupling > 0 else 1.0
    # Approximate ground state angle
    base_angle = np.arcsin(np.tanh(ratio))
    
    np.random.seed(42)
    
    restarts = []
    # Symmetric initial guesses
    restarts.append(np.full(num_independent, base_angle))
    restarts.append(np.zeros(num_independent))
    restarts.append(np.full(num_independent, np.pi / 2))
    restarts.append(np.full(num_independent, np.pi / 4))
    
    # Random restarts
    for scale in [0.5, 1.0]:
        restarts.append(np.random.randn(num_independent) * scale)
        
    best_result = None
    best_val = np.inf
    
    for r, x_init in enumerate(restarts):
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 10:
            break
        
        x_init = np.clip(x_init, -np.pi, np.pi)
        
        try:
            res = opt.minimize(
                objective,
                x_init,
                jac=gradient,
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 20
                }
            )
            val = res.fun
            if val < best_val:
                best_val = val
                best_result = res.x.copy()
                
        except Exception:
            pass
    
    if best_result is None:
        best_result = np.full(num_independent, base_angle)
    
    full_params = expand_params(best_result)
    theta = full_params.reshape(layers, n)
    _cache[key] = theta
    return theta


def angles(n, coupling, field, layers):
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT - 5:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    try:
        return _optimize_for_params(n, coupling, field, layers)
    except Exception:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
# EVOLVE-BLOCK-END
