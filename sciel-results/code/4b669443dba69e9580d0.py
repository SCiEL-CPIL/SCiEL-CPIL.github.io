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
    params: array of 12 floats representing a_half and b_half.
    a_half = params[:6], b_half = params[6:12]
    
    The full coefficients are constructed by symmetry:
    a_full = [a_1, a_2, a_3, a_4, a_5, a_6, a_6, a_5, a_4, a_3, a_2, a_1]
    Sum(a_full) = 2 * sum(a_1..a_6).
    We normalize so that sum(a_full) = 1.
    Same for b.
    """
    a_half = params[:6]
    b_half = params[6:12]
    
    # Construct symmetric full lists
    a_full_raw = np.concatenate([a_half, a_half[::-1]])
    b_full_raw = np.concatenate([b_half, b_half[::-1]])
    
    # Normalize to ensure sum is exactly 1
    sum_a = np.sum(a_full_raw)
    sum_b = np.sum(b_full_raw)
    
    # Avoid division by zero
    if np.abs(sum_a) < 1e-12:
        sum_a = 1e-12
    if np.abs(sum_b) < 1e-12:
        sum_b = 1e-12
        
    a_full = a_full_raw / sum_a
    b_full = b_full_raw / sum_b
    
    # Clip to valid range to prevent numerical explosion in exp
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
    
    # Pre-computed eigendecompositions:
    # A = V_A diag(lambda_A) V_A^H
    # exp(-i c t A) = V_A diag(exp(-i c t lambda_A)) V_A^H
    V_A, lam_A = eig_A
    V_B, lam_B = eig_B
    
    V_A_h = V_A.conj().T
    V_B_h = V_B.conj().T
    
    for i, t in enumerate(t_values):
        U = np.eye(dim, dtype=complex)
        
        for k in range(12):
            # exp(-i b_k t B)
            exp_B = V_B @ np.diag(np.exp(-1j * b_full[k] * t * lam_B)) @ V_B_h
            # exp(-i a_k t A)
            exp_A = V_A @ np.diag(np.exp(-i * a_full[k] * t * lam_A)) @ V_A_h
            
            # Order: exp(-i b_k t B) exp(-i a_k t A)
            U = exp_B @ exp_A @ U
        
        diff = U - exacts[i]
        # Spectral norm is the largest singular value
        svals = np.linalg.svd(diff, compute_uv=False)
        total_error += svals[0]
        
    return total_error / len(t_values)

def _nelder_mead(func, x0, args=(), max_iter=500, timeout=None, start_time=None):
    """
    Pure NumPy implementation of Nelder-Mead simplex optimization.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(x0)
    # Initial simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        point = np.array(x0, dtype=float)
        step = 0.05 if x0[i] == 0 else 0.05 * abs(x0[i])
        point[i] += step
        simplex.append(point)
    
    values = [func(p, *args) for p in simplex]
    best_val = min(values)
    best_point = simplex[values.index(best_val)]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for iteration in range(max_iter):
        if timeout is not None and time.time() - start_time > timeout:
            break
            
        # Sort simplex by function value
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[0] - values[-1]) < 1e-9:
            break
            
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        worst_point = simplex[-1]
        reflected = centroid + alpha * (centroid - worst_point)
        f_reflected = func(reflected, *args)
        
        if f_reflected < values[0]:
            # Expansion
            expanded = centroid + gamma * (reflected - centroid)
            f_expanded = func(expanded, *args)
            if f_expanded < f_reflected:
                simplex[-1] = expanded
                values[-1] = f_expanded
            else:
                simplex[-1] = reflected
                values[-1] = f_reflected
        elif f_reflected < values[-2]:
            simplex[-1] = reflected
            values[-1] = f_reflected
        else:
            # Contraction
            contracted = centroid + rho * (worst_point - centroid)
            f_contracted = func(contracted, *args)
            if f_contracted < values[-1]:
                simplex[-1] = contracted
                values[-1] = f_contracted
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    values[i] = func(simplex[i], *args)
        
        # Update best
        current_best_val = min(values)
        if current_best_val < best_val:
            best_val = current_best_val
            best_point = simplex[values.index(current_best_val)]
            
    return best_point, best_val

def coefficients(stages):
    start_time = time.time()
    
    # 1. Build Operators
    A, B = _build_operators()
    
    # 2. Pre-compute Eigendecompositions for A and B
    # A and B are Hermitian
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eig_data_A = (eigvecs_A, eigvals_A)
    
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    eig_data_B = (eigvecs_B, eigvals_B)
    
    # 3. Compute Exact Propagators
    t_values = [0.25, 0.5, 1.0]
    exacts = _compute_exact_propagators(A, B, t_values)
    
    # 4. Initial Guess
    # Uniform 2nd-order symmetric splitting: each coeff = 1/12
    # a_full = [1/12, ..., 1/12]
    # a_half = [1/12, ..., 1/12] (6 elements)
    # Sum(a_half) = 6/12 = 0.5. Normalized sum = 2*0.5=1. Correct.
    a_init = np.full(6, 1.0/12.0)
    b_init = np.full(6, 1.0/12.0)
    x0 = np.concatenate([a_init, b_init])
    
    # 5. Optimization
    best_params = x0
    best_err = _eval_error(x0, A, B, eig_data_A, eig_data_B, exacts, t_values)
    
    try:
        # Run Nelder-Mead with time limit
        # We allocate most of the time to the first run
        time_limit_1 = _DEADLINE - (time.time() - start_time) - 5.0
        if time_limit_1 > 10.0:
            p1, v1 = _nelder_mead(
                _eval_error, 
                x0, 
                args=(A, B, eig_data_A, eig_data_B, exacts, t_values),
                max_iter=2000,
                timeout=time_limit_1,
                start_time=start_time
            )
            if v1 < best_err:
                best_err = v1
                best_params = p1
        
        # Check time for a second refinement run
        elapsed = time.time() - start_time
        if elapsed < _DEADLINE - 15:
            time_limit_2 = _DEADLINE - elapsed - 5.0
            p2, v2 = _nelder_mead(
                _eval_error, 
                best_params, 
                args=(A, B, eig_data_A, eig_data_B, exacts, t_values),
                max_iter=1000,
                timeout=time_limit_2,
                start_time=start_time
            )
            if v2 < best_err:
                best_err = v2
                best_params = p2
                
    except Exception:
        # If optimization fails, keep the initial guess
        pass
    
    # 6. Extract Final Coefficients
    a_half = best_params[:6]
    b_half = best_params[6:12]
    
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])
    
    # Ensure sums are exactly 1 (correct for any numerical drift)
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
