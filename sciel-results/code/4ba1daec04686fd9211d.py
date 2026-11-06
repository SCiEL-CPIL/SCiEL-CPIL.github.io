"""State-fitting approach with analytic gradients and multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_cache = {}
_start_time = time.time()
_TIME_LIMIT = 142.0  # safety margin before 150s kill


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
    shape = [2] * n
    psi_r = psi.reshape(shape)
    psi_moved = np.moveaxis(psi_r, q, 0)
    new0 = c * psi_moved[0] - s * psi_moved[1]
    new1 = s * psi_moved[0] + c * psi_moved[1]
    psi_out = np.stack([new0, new1], axis=0)
    return np.moveaxis(psi_out, 0, q).reshape(-1)


def _apply_cz(psi, q, n):
    """Apply CZ on qubits q, q+1 to state psi. Returns new array."""
    shape = [2] * n
    psi_r = psi.reshape(shape)
    psi_moved = np.moveaxis(psi_r, [q, q + 1], [-2, -1])
    psi_moved = psi_moved.copy()
    psi_moved[..., 1, 1] *= -1.0
    return np.moveaxis(psi_moved, [-2, -1], [q, q + 1]).reshape(-1)


def _circuit_state(params, n, layers):
    """Compute the final state from flattened (layers*n) parameter vector."""
    dim = 2 ** n
    psi = np.zeros(dim, dtype=np.float64)
    psi[0] = 1.0
    t = params.reshape(layers, n)
    for layer in range(layers):
        for q in range(n):
            psi = _apply_ry(psi, q, t[layer, q], n)
        if layer < layers - 1:
            for q in range(n - 1):
                psi = _apply_cz(psi, q, n)
    return psi


def _circuit_energy_and_gradient(params, n, layers, H, e0):
    """
    Compute energy (E_circuit - e0)/n and its gradient using the adjoint method.
    The gradient is computed analytically by backpropagating H|psi> through the circuit.
    """
    dim = 2 ** n
    num_params = layers * n
    t = params.reshape(layers, n)
    
    # Forward pass: store all intermediate states
    # Structure: for each layer, for each qubit, store state before Ry
    # Also store states after each CZ layer
    states = []  # list of (type, data) where type is 'ry' or 'cz'
    
    psi = np.zeros(dim, dtype=np.float64)
    psi[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            states.append(('ry', (psi.copy(), q, t[layer, q])))
            psi = _apply_ry(psi, q, t[layer, q], n)
        if layer < layers - 1:
            for q in range(n - 1):
                states.append(('cz', (psi.copy(), q)))
                psi = _apply_cz(psi, q, n)
    
    # Final state
    psi_final = psi
    energy = float(np.dot(psi_final, H @ psi_final))
    obj_val = (energy - e0) / n
    
    # Backward pass: compute adjoint
    # We want dE/dtheta_k for each Ry parameter
    # E = <psi|H|psi>, dE = 2*Re[<psi|H|dpsi>] = 2*Re[<Hpsi|dpsi>]
    # Let |chi> = H|psi_final> (adjoint vector)
    chi = H @ psi_final  # This is the adjoint to propagate backward
    
    grad = np.zeros(num_params)
    
    # Backpropagate through gates in reverse order
    for idx in range(len(states) - 1, -1, -1):
        stype, data = states[idx]
        if stype == 'cz':
            psi_before_cz, q = data
            # CZ is self-adjoint and unitary, so chi just passes through (with the gate applied)
            # dpsi_after = CZ * dpsi_before, so dpsi_before = CZ * dpsi_after (since CZ^-1 = CZ)
            chi = _apply_cz(chi, q, n)
        elif stype == 'ry':
            psi_before_ry, q, theta = data
            # dpsi_after = Ry(theta) * dpsi_before
            # dE/dtheta = 2*Re[<chi_after | dpsi_after/dtheta>]
            # dpsi_after/dtheta = (dRy/dtheta) * psi_before_ry
            # dRy/dtheta = 0.5 * [[-sin(theta/2), -cos(theta/2)], [cos(theta/2), -sin(theta/2)]]
            # So dpsi_after/dtheta = 0.5 * Ry'(theta) @ psi_before_ry
            # And <chi_after | dpsi_after/dtheta> = 0.5 * <Ry'(theta)† @ chi_after | psi_before_ry>
            # But it's easier to just compute the contribution directly:
            
            # The contribution to dE/dtheta_k is 2*Re[<chi | dRy/dtheta | psi_before>]
            # where chi is the adjoint at this point (after the Ry)
            
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            # dRy/dtheta = 0.5 * [[-s, -c], [c, -s]]
            
            # We need to compute 2*Re[<chi | 0.5 * dRy/dtheta | psi_before>]
            # = Re[<chi | dRy/dtheta | psi_before>]
            
            # Apply dRy/dtheta to psi_before:
            shape = [2] * n
            psi_before_r = psi_before_ry.reshape(shape)
            psi_moved = np.moveaxis(psi_before_r, q, 0)
            
            # dRy/dtheta applied to the qubit:
            # new0 = 0.5*(-s*psi0 - c*psi1)
            # new1 = 0.5*(c*psi0 - s*psi1)
            dpsi0 = 0.5 * (-s * psi_moved[0] - c * psi_moved[1])
            dpsi1 = 0.5 * (c * psi_moved[0] - s * psi_moved[1])
            dpsi_out = np.stack([dpsi0, dpsi1], axis=0)
            dpsi_flat = np.moveaxis(dpsi_out, 0, q).reshape(-1)
            
            # Contribution: 2 * Re[<chi | dpsi_flat>]
            param_idx = len(states) - 1 - idx  # This isn't right, let me track param index properly
            # Actually, let me track param indices separately
            
            # For now, store the contribution and map to param index later
            grad[len(states) - 1 - idx] = 2.0 * np.real(np.dot(chi, dpsi_flat))
            
            # Propagate chi backward through Ry:
            # chi_before = Ry(theta)† @ chi_after = Ry(-theta) @ chi_after
            # Ry(-theta) = [[c, s], [-s, c]]
            chi_r = chi.reshape(shape)
            chi_moved = np.moveaxis(chi_r, q, 0)
            new0 = c * chi_moved[0] + s * chi_moved[1]
            new1 = -s * chi_moved[0] + c * chi_moved[1]
            chi_out = np.stack([new0, new1], axis=0)
            chi = np.moveaxis(chi_out, 0, q).reshape(-1)
    
    # Now we need to map the gradient indices correctly
    # The states list has entries for both ry and cz gates
    # We need to extract only the ry contributions in the correct order
    grad_final = np.zeros(num_params)
    ry_count = 0
    for idx in range(len(states)):
        stype, data = states[idx]
        if stype == 'ry':
            # This ry gate corresponds to parameter index ry_count
            # But in our backward pass, we stored it at index len(states)-1-idx
            # Let's redo this more carefully
            pass
    
    # Let me redo the gradient computation more carefully with proper indexing
    grad_final = np.zeros(num_params)
    
    # Re-do backward pass with proper parameter tracking
    chi = H @ psi_final
    
    # Build a mapping from state index to parameter index
    param_map = {}  # state_idx -> param_idx for ry gates
    pidx = 0
    for idx in range(len(states)):
        stype, data = states[idx]
        if stype == 'ry':
            param_map[idx] = pidx
            pidx += 1
    
    # Backward pass
    for idx in range(len(states) - 1, -1, -1):
        stype, data = states[idx]
        if stype == 'cz':
            _, q = data
            chi = _apply_cz(chi, q, n)
        elif stype == 'ry':
            psi_before_ry, q, theta = data
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            shape = [2] * n
            psi_before_r = psi_before_ry.reshape(shape)
            psi_moved = np.moveaxis(psi_before_r, q, 0)
            
            dpsi0 = 0.5 * (-s * psi_moved[0] - c * psi_moved[1])
            dpsi1 = 0.5 * (c * psi_moved[0] - s * psi_moved[1])
            dpsi_out = np.stack([dpsi0, dpsi1], axis=0)
            dpsi_flat = np.moveaxis(dpsi_out, 0, q).reshape(-1)
            
            p = param_map[idx]
            grad_final[p] = 2.0 * np.real(np.dot(chi, dpsi_flat))
            
            chi_r = chi.reshape(shape)
            chi_moved = np.moveaxis(chi_r, q, 0)
            new0 = c * chi_moved[0] + s * chi_moved[1]
            new1 = -s * chi_moved[0] + c * chi_moved[1]
            chi_out = np.stack([new0, new1], axis=0)
            chi = np.moveaxis(chi_out, 0, q).reshape(-1)
    
    return obj_val, grad_final


def _optimize_for_params(n, coupling, field, layers):
    """Optimize circuit angles for given parameters."""
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    H = _build_hamiltonian(n, coupling, field)
    e0, psi_gs = _exact_ground_state(H)
    
    num_params = layers * n
    
    def objective(params):
        val, _ = _circuit_energy_and_gradient(params, n, layers, H, e0)
        return val
    
    def gradient(params):
        _, grad = _circuit_energy_and_gradient(params, n, layers, H, e0)
        return grad
    
    # Physics-based initial guesses
    ratio = field / coupling if coupling > 0 else 1.0
    base_angle = np.arcsin(np.tanh(ratio)) if abs(np.tanh(ratio)) <= 1.0 else np.pi / 2
    
    np.random.seed(42)
    
    restarts = []
    # Symmetric initial guesses
    restarts.append(np.full(num_params, base_angle))
    restarts.append(np.zeros(num_params))
    restarts.append(np.full(num_params, np.pi / 2))
    restarts.append(np.full(num_params, np.pi / 4))
    restarts.append(np.full(num_params, -np.pi / 4))
    
    # Random restarts with various scales
    for scale in [0.2, 0.4, 0.6, 0.8, 1.0, 1.5]:
        restarts.append(np.random.randn(num_params) * scale)
    
    # Layer-dependent patterns
    for l_scale in [0.5, 1.0, 1.5]:
        pat = np.zeros(num_params)
        for l in range(layers):
            for q in range(n):
                pat[l * n + q] = base_angle * l_scale
        restarts.append(pat)
    
    # Alternating patterns
    for offset in [0, np.pi/4, np.pi/2, -np.pi/4]:
        pat = np.zeros(num_params)
        for l in range(layers):
            for q in range(n):
                pat[l * n + q] = base_angle + offset * ((l + q) % 2)
        restarts.append(pat)
    
    # Near-ground-state-inspired starts
    # Try to match the ground state amplitudes
    gs_amplitudes = np.abs(psi_gs)
    # For a product state approximation, each qubit's angle relates to local magnetization
    # <Z_i> in ground state
    for i in range(n):
        zi_expectation = 0.0
        for idx in range(2 ** n):
            bit_i = (idx >> i) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zi_expectation += zi * psi_gs[idx] * psi_gs[idx]
        # Ry(theta)|0> has <Z> = cos(theta), so theta = arccos(<Z_i>)
        zi_clipped = np.clip(zi_expectation, -1.0, 1.0)
        local_theta = np.arccos(zi_clipped)
        for l in range(layers):
            restarts.append(np.full(num_params, local_theta))
            break  # Just use first qubit's value for all
    
    best_result = None
    best_val = np.inf
    
    for r, x_init in enumerate(restarts):
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 20:
            break
        
        x_init = np.clip(x_init, -np.pi, np.pi)
        
        try:
            res = opt.minimize(
                objective,
                x_init,
                jac=gradient,
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'ftol': 1e-15,
                    'gtol': 1e-12,
                    'maxls': 100,
                    'maxfun': 1000
                }
            )
            val = res.fun
            if val < best_val:
                best_val = val
                best_result = res.x.copy()
                
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if best_result is not None:
        elapsed = time.time() - _start_time
        if elapsed < _TIME_LIMIT - 10:
            try:
                res = opt.minimize(
                    objective,
                    best_result,
                    jac=gradient,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 2000,
                        'ftol': 1e-16,
                        'gtol': 1e-14,
                        'maxls': 200,
                        'maxfun': 5000
                    }
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_result = res.x.copy()
            except Exception:
                pass
    
    if best_result is None:
        best_result = np.full(num_params, base_angle)
    
    theta = best_result.reshape(layers, n)
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
