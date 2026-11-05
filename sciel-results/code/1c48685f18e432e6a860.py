import time
import numpy as np

_DEADLINE = 140.0  # Safety margin before 150s kill

def _build_operators(n_sites=6, anisotropy=0.8, field=0.3):
    """
    Build A (odd bonds + half field) and B (even bonds + half field) for a 6-site
    open XXZ Heisenberg chain.
    """
    dim = 2 ** n_sites
    
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)

    def kron_all(ops_list):
        result = np.array([[1.0]], dtype=complex)
        for op in ops_list:
            result = np.kron(result, op)
        return result

    def bond_operator(j, gamma, h_half):
        ops = [identity] * n_sites
        ops[j] = sx
        ops[j + 1] = sx
        term_xx = kron_all(ops)

        ops = [identity] * n_sites
        ops[j] = sy
        ops[j + 1] = sy
        term_yy = kron_all(ops)

        ops = [identity] * n_sites
        ops[j] = sz
        ops[j + 1] = sz
        term_zz = gamma * kron_all(ops)

        ops = [identity] * n_sites
        ops[j] = sz
        term_field_j = (-h_half) * kron_all(ops)

        ops = [identity] * n_sites
        ops[j + 1] = sz
        term_field_j1 = (-h_half) * kron_all(ops)

        return term_xx + term_yy + term_zz + term_field_j + term_field_j1

    A = np.zeros((dim, dim), dtype=complex)
    for j in [0, 2, 4]:
        A += bond_operator(j, anisotropy, field / 2)

    B = np.zeros((dim, dim), dtype=complex)
    for j in [1, 3]:
        B += bond_operator(j, anisotropy, field / 2)

    return A, B

def _compute_exact_propagators(A, B, t_values):
    """Compute exact propagators exp(-i t (A+B)) for given t values."""
    H = A + B
    eigvals, eigvecs = np.linalg.eigh(H)
    exacts = []
    for t in t_values:
        exact = eigvecs @ np.diag(np.exp(-1j * t * eigvals)) @ eigvecs.conj().T
        exacts.append(exact)
    return exacts

def _eval_error(params, A, B, eig_A, eig_B, exacts, t_values):
    """
    params: array of 12 floats representing log-space coefficients.
    We use a symmetric parameterization where:
    a_full = [s1, s2, s3, s4, s5, s6, s6, s5, s4, s3, s2, s1]
    Similarly for b.
    The params are in log-space to ensure positivity.
    """
    a_half_log = params[:6]
    b_half_log = params[6:12]
    
    # Convert from log-space to actual values
    a_half = np.exp(a_half_log)
    b_half = np.exp(b_half_log)
    
    # Construct symmetric full lists
    a_full_raw = np.concatenate([a_half, a_half[::-1]])
    b_full_raw = np.concatenate([b_half, b_half[::-1]])
    
    # Normalize to ensure sum is exactly 1
    sum_a = np.sum(a_full_raw)
    sum_b = np.sum(b_full_raw)
    
    if np.abs(sum_a) < 1e-12:
        sum_a = 1e-12
    if np.abs(sum_b) < 1e-12:
        sum_b = 1e-12
        
    a_full = a_full_raw / sum_a
    b_full = b_full_raw / sum_b
    
    # Clip to valid range
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)
    
    # Re-normalize after clipping if necessary
    sum_a_clip = np.sum(a_full)
    sum_b_clip = np.sum(b_full)
    if np.abs(sum_a_clip) > 1e-12:
        a_full = a_full / sum_a_clip
    if np.abs(sum_b_clip) > 1e-12:
        b_full = b_full / sum_b_clip

    total_error = 0.0
    dim = A.shape[0]
    
    V_A, lam_A = eig_A
    V_B, lam_B = eig_B
    
    V_A_h = V_A.conj().T
    V_B_h = V_B.conj().T
    
    for i, t in enumerate(t_values):
        U = np.eye(dim, dtype=complex)
        
        for k in range(12):
            exp_B = V_B @ np.diag(np.exp(-1j * b_full[k] * t * lam_B)) @ V_B_h
            exp_A = V_A @ np.diag(np.exp(-1j * a_full[k] * t * lam_A)) @ V_A_h
            U = exp_B @ exp_A @ U
        
        diff = U - exacts[i]
        svals = np.linalg.svd(diff, compute_uv=False)
        total_error += svals[0]
        
    return total_error / len(t_values)

def _eval_error_with_grad(params, A, B, eig_A, eig_B, exacts, t_values):
    """
    Evaluate error and approximate gradient using finite differences.
    """
    f0 = _eval_error(params, A, B, eig_A, eig_B, exacts, t_values)
    grad = np.zeros_like(params)
    eps = 1e-6
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _eval_error(params_plus, A, B, eig_A, eig_B, exacts, t_values)
        grad[i] = (f_plus - f0) / eps
    return f0, grad

def _lbfgs_b(func, x0, args=(), maxiter=500, timeout=None, start_time=None):
    """
    Simple L-BFGS-B implementation using scipy if available, else fallback.
    """
    try:
        from scipy.optimize import minimize
        result = minimize(
            func, x0, args=args, method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return result.x, result.fun
    except ImportError:
        # Fallback to a simple gradient descent
        x = np.array(x0, dtype=float)
        best_x = x.copy()
        best_f = func(x, *args)
        
        for iteration in range(maxiter):
            if timeout is not None and time.time() - start_time > timeout:
                break
            
            f, grad = _eval_error_with_grad(x, *args)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            
            # Adaptive step size
            lr = 0.01 / (1 + iteration * 0.01)
            x = x - lr * grad
            x = np.clip(x, -10, 10)  # Keep log-space values reasonable
        
        return best_x, best_f

def coefficients(stages):
    start_time = time.time()
    
    # 1. Build Operators
    A, B = _build_operators()
    
    # 2. Pre-compute Eigendecompositions for A and B
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eig_data_A = (eigvecs_A, eigvals_A)
    
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    eig_data_B = (eigvecs_B, eigvals_B)
    
    # 3. Compute Exact Propagators
    t_values = [0.25, 0.5, 1.0]
    exacts = _compute_exact_propagators(A, B, t_values)
    
    # 4. Initial Guess
    # Start with uniform coefficients in log-space
    # We want a_full to sum to 1, so each element should be 1/12
    # a_half has 6 elements, each contributing twice, so each should be 1/12
    # In log-space: log(1/12)
    init_val = np.log(1.0 / 12.0)
    x0 = np.full(12, init_val)
    
    # 5. Multi-stage optimization
    best_params = x0.copy()
    best_err = _eval_error(x0, A, B, eig_data_A, eig_data_B, exacts, t_values)
    
    try:
        # Stage 1: Coarse optimization with limited iterations
        elapsed = time.time() - start_time
        time_remaining = _DEADLINE - elapsed - 20.0
        if time_remaining > 10.0:
            p1, v1 = _lbfgs_b(
                _eval_error, 
                x0, 
                args=(A, B, eig_data_A, eig_data_B, exacts, t_values),
                maxiter=100,
                timeout=time_remaining,
                start_time=start_time
            )
            if v1 < best_err:
                best_err = v1
                best_params = p1
        
        # Stage 2: Fine optimization
        elapsed = time.time() - start_time
        time_remaining = _DEADLINE - elapsed - 10.0
        if time_remaining > 5.0:
            p2, v2 = _lbfgs_b(
                _eval_error, 
                best_params, 
                args=(A, B, eig_data_A, eig_data_B, exacts, t_values),
                maxiter=200,
                timeout=time_remaining,
                start_time=start_time
            )
            if v2 < best_err:
                best_err = v2
                best_params = p2
                
    except Exception:
        # If optimization fails, keep the initial guess
        pass
    
    # 6. Extract Final Coefficients
    a_half_log = best_params[:6]
    b_half_log = best_params[6:12]
    
    a_half = np.exp(a_half_log)
    b_half = np.exp(b_half_log)
    
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])
    
    # Ensure sums are exactly 1
    sum_a = np.sum(a_full)
    sum_b = np.sum(b_full)
    
    if abs(sum_a) > 1e-12:
        a_full = a_full / sum_a
    if abs(sum_b) > 1e-12:
        b_full = b_full / sum_b
        
    # Clip to bounds
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)
    
    # Re-normalize after clipping
    sum_a = np.sum(a_full)
    sum_b = np.sum(b_full)
    if abs(sum_a) > 1e-12:
        a_full = a_full / sum_a
    if abs(sum_b) > 1e-12:
        b_full = b_full / sum_b
        
    return list(a_full), list(b_full)
