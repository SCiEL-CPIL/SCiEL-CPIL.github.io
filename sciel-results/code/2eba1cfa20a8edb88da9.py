import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

def _compute_c_eigenvalues(n, edges):
    """Compute eigenvalues of C = sum_{(i,j) in E} (1 - Z_i Z_j)/2 for each basis state."""
    num_states = 1 << n
    c_vals = np.zeros(num_states)
    for k in range(num_states):
        c_val = 0.0
        for (i, j) in edges:
            bit_i = (k >> i) & 1
            bit_j = (k >> j) & 1
            if bit_i == bit_j:
                c_val += 0.0  # (1 - 1)/2 = 0
            else:
                c_val += 1.0  # (1 - (-1))/2 = 1
        c_vals[k] = c_val
    return c_vals

def _apply_beta_rotation(state, beta):
    """Apply exp(-i beta * sum X_i) = tensor product of single-qubit rotations."""
    # exp(-i beta X) = cos(beta) I - i sin(beta) X
    # In Z-basis: [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
    # state shape: (2^n,) = tensor product of n qubits
    n = int(np.log2(len(state)))
    state = state.reshape([2] * n)
    c = np.cos(beta)
    s = -1j * np.sin(beta)
    # Apply to each qubit: new_state[i0,...,ik,...,in] = c*old[i0,...,ik,...,in] + s*old[i0,...,1-ik,...,in]
    for q in range(n):
        new_state = np.zeros_like(state)
        for val in range(2):
            # Create index arrays
            idx = [slice(None)] * n
            idx[q] = val
            # The flipped index
            idx_flip = [slice(None)] * n
            idx_flip[q] = 1 - val
            new_state[tuple(idx)] += c * state[tuple(idx)] + s * state[tuple(idx_flip)]
        state = new_state
    return state.reshape(-1)

def _apply_gamma_rotation(state, gamma, c_vals):
    """Apply exp(-i gamma * C) which is diagonal in Z-basis."""
    return state * np.exp(-1j * gamma * c_vals)

def _expected_cut_and_grad(n, edges, angles, c_vals):
    """Compute expected cut and its gradient w.r.t. angles.
    angles: array of shape (p, 2), rows are (gamma, beta)
    Returns: (expected_cut, grad) where grad has shape (p, 2)
    """
    p = angles.shape[0]
    num_states = 1 << n
    state = np.ones(num_states, dtype=complex) / np.sqrt(num_states)
    
    # Forward pass: save intermediate states
    states = [state.copy()]
    for l in range(p):
        gamma, beta = angles[l]
        state = _apply_gamma_rotation(state, gamma, c_vals)
        states.append(state.copy())
        state = _apply_beta_rotation(state, beta)
        states.append(state.copy())
    
    # Compute <Z_i Z_j> for each edge
    expected_cut = 0.0
    for (i, j) in edges:
        zz = np.zeros(num_states)
        for k in range(num_states):
            bit_i = (k >> i) & 1
            bit_j = (k >> j) & 1
            zz[k] = 1.0 if bit_i == bit_j else -1.0
        expected_cut += (1.0 - np.real(np.vdot(state, zz * state))) / 2.0
    
    # Backward pass for gradients
    # We need d/dparams of <psi|C|psi> where C is the cost operator
    # Actually, let's compute gradient using the fact that:
    # d/dgamma_l <C> = -i * <psi| [C, exp(i gamma_l C_l) ... ] |psi> ... this is complex
    # 
    # Alternative: use parameter-shift rule or just finite differences for gradient
    # Given the small size, let's use analytic gradient via chain rule
    #
    # Actually, let me use a different approach: compute gradient via 
    # d/dtheta <psi|O|psi> = 2 Re(<dpsi/dtheta|O|psi>)
    #
    # For gamma: dpsi/dgamma_l involves the derivative of the unitary
    # For beta: dpsi/dbeta_l involves the derivative of the single-qubit rotation
    
    # Let me use a simpler approach: since the state is small, use 
    # complex-step or just compute gradients analytically
    
    # Actually, for efficiency, let me precompute the Z_i Z_j vectors
    # and use the chain rule properly
    
    # Let's redo this with proper backpropagation
    # The objective is f = <psi|C|psi> where C is diagonal with entries c_vals
    # So f = sum_k |psi_k|^2 * c_vals[k]
    
    # Backprop through the circuit
    # d f / d psi* = C * psi (element-wise: c_vals * psi)
    # d f / d psi = C * psi* (conjugate)
    # But since f is real, d f / d psi* = c_vals * psi
    
    # For each layer, we need to propagate the gradient back
    
    # Let me define: 
    # state_{l+1} = U_beta_l (U_gamma_l (state_l))
    # where state_0 = |+>^n
    
    # Gradient w.r.t. gamma_l:
    # d state_{l+1} / d gamma_l = d U_beta_l / d gamma_l * ... no, U_beta doesn't depend on gamma
    # d state_{l+1} / d gamma_l = U_beta_l * d U_gamma_l / d gamma_l * state_l
    # d U_gamma_l / d gamma_l = -i * C * U_gamma_l (since U_gamma = exp(-i gamma C))
    # So d state_{l+1} / d gamma_l = U_beta_l * (-i * C * U_gamma_l) * state_l
    # = -i * U_beta_l * C * state_l (since U_gamma_l * state_l is the state after gamma)
    # Let me call state_after_gamma_l = U_gamma_l * state_l
    # Then d state_{l+1} / d gamma_l = -i * U_beta_l * (C * state_after_gamma_l)
    
    # Gradient w.r.t. beta_l:
    # d state_{l+1} / d beta_l = d U_beta_l / d beta_l * state_after_gamma_l
    # U_beta_l = (R_x(beta))^⊗n where R_x(beta) = exp(-i beta X)
    # d R_x / d beta = -i X R_x
    # So d U_beta / d beta = sum_k (I ⊗ ... ⊗ (-i X) ⊗ ... ⊗ I) * U_beta
    # = -i * (sum_k X_k) * U_beta = -i * B * U_beta where B = sum X_k
    
    # This is getting complex. Let me just use the parameter-shift rule for simplicity
    # and since n is small, it's fast enough.
    
    # Actually, let me just compute the gradient numerically with central differences
    # for the 4 parameters. This is 8 evaluations per gradient, which is fine.
    
    grad = np.zeros((p, 2))
    eps = 1e-7
    
    # Helper to compute expected cut for given angles
    def _eval(angles_eval):
        st = np.ones(num_states, dtype=complex) / np.sqrt(num_states)
        for l in range(p):
            g, b = angles_eval[l]
            st = _apply_gamma_rotation(st, g, c_vals)
            st = _apply_beta_rotation(st, b)
        ec = 0.0
        for (i, j) in edges:
            zz = np.zeros(num_states)
            for k in range(num_states):
                bit_i = (k >> i) & 1
                bit_j = (k >> j) & 1
                zz[k] = 1.0 if bit_i == bit_j else -1.0
            ec += (1.0 - np.real(np.vdot(st, zz * st))) / 2.0
        return ec
    
    # This is too slow with the inner loop. Let me precompute the ZZ vectors.
    
    return expected_cut, grad

def _expected_cut_fast(n, edges, angles, c_vals, zz_vectors):
    """Compute expected cut efficiently using precomputed ZZ vectors."""
    p = angles.shape[0]
    num_states = 1 << n
    state = np.ones(num_states, dtype=complex) / np.sqrt(num_states)
    
    for l in range(p):
        gamma, beta = angles[l]
        state = _apply_gamma_rotation(state, gamma, c_vals)
        state = _apply_beta_rotation(state, beta)
    
    expected_cut = 0.0
    for (i, j) in edges:
        zz = zz_vectors[(i, j)]
        expected_cut += (1.0 - np.real(np.vdot(state, zz * state))) / 2.0
    
    return expected_cut

def _build_zz_vectors(n, edges):
    """Precompute Z_i Z_j vectors for all edges."""
    num_states = 1 << n
    zz_vectors = {}
    for (i, j) in edges:
        zz = np.zeros(num_states)
        for k in range(num_states):
            bit_i = (k >> i) & 1
            bit_j = (k >> j) & 1
            zz[k] = 1.0 if bit_i == bit_j else -1.0
        zz_vectors[(i, j)] = zz
    return zz_vectors

def _exact_optimum(n, edges):
    """Compute exact MaxCut by exhaustive search."""
    num_states = 1 << n
    best = 0
    for k in range(num_states):
        cut = 0
        for (i, j) in edges:
            bit_i = (k >> i) & 1
            bit_j = (k >> j) & 1
            if bit_i != bit_j:
                cut += 1
        best = max(best, cut)
    return best

def angles(n, edges, p):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    edges = list(edges)
    c_vals = _compute_c_eigenvalues(n, edges)
    zz_vectors = _build_zz_vectors(n, edges)
    opt_cut = _exact_optimum(n, edges)
    
    def objective(angles_flat):
        angles_arr = angles_flat.reshape(p, 2)
        return -_expected_cut_fast(n, edges, angles_arr, c_vals, zz_vectors)
    
    def objective_and_grad(angles_flat):
        angles_arr = angles_flat.reshape(p, 2)
        val = -_expected_cut_fast(n, edges, angles_arr, c_vals, zz_vectors)
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(angles_flat)
        for i in range(len(angles_flat)):
            orig = angles_flat[i]
            angles_flat[i] = orig + eps
            f_plus = -_expected_cut_fast(n, edges, angles_flat.reshape(p, 2), c_vals, zz_vectors)
            angles_flat[i] = orig - eps
            f_minus = -_expected_cut_fast(n, edges, angles_flat.reshape(p, 2), c_vals, zz_vectors)
            angles_flat[i] = orig
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return val, grad
    
    best_result = None
    best_val = -np.inf
    
    # Multiple restarts from diverse initial points
    np.random.seed(42)
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Generate initial guess
        if restart == 0:
            init = np.array([0.5, 0.3, 0.3, 0.2])
        elif restart == 1:
            init = np.array([0.35, 0.25, 0.35, 0.25])
        else:
            init = np.random.uniform(0, np.pi, size=4)
        
        # L-BFGS-B optimization
        try:
            result = minimize(objective, init, method='L-BFGS-B', 
                            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.success or result.fun < best_val:
                if result.fun < best_val:
                    best_val = result.fun
                    best_result = result.x.copy()
        except Exception:
            pass
        
        # Nelder-Mead polish
        if best_result is not None and time.time() < deadline - 10.0:
            try:
                result_nm = minimize(objective, best_result, method='Nelder-Mead',
                                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12})
                if result_nm.fun < best_val:
                    best_val = result_nm.fun
                    best_result = result_nm.x.copy()
            except Exception:
                pass
    
    if best_result is None:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Reshape to (p, 2)
    return best_result.reshape(p, 2)
