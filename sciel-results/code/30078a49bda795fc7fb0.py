import time
import numpy as np
from scipy.optimize import minimize

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
    eigvals, eigvecs = np.linalg.eigh(H) # Use eigh for Hermitian matrices
    exacts = []
    for t in t_values:
        # U = V @ diag(exp(-i t lambda)) @ V^H
        # Since V is unitary for Hermitian H, V^-1 = V^H
        exact = eigvecs @ np.diag(np.exp(-1j * t * eigvals)) @ eigvecs.conj().T
        exacts.append(exact)
    return exacts

def _eval_error(params, A, B, eig_A, eig_B, exacts, t_values):
    """
    params: array of 12 floats.
    First 6: a_1..a_6 (first half of A coefficients)
    Last 6: b_1..b_6 (first half of B coefficients)
    
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
    # Note: Clipping after normalization might break the sum=1 constraint slightly,
    # but the values are expected to be small. If clipping occurs, we re-normalize.
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
            # diag_B = exp(-i b_k t lam_B)
            exp_B = V_B @ np.diag(np.exp(-1j * b_full[k] * t * lam_B)) @ V_B_h
            # exp(-i a_k t A)
            exp_A = V_A @ np.diag(np.exp(-1j * a_full[k] * t * lam_A)) @ V_A_h
            
            # Order: exp(-i b_k t B) exp(-i a_k t A)
            U = exp_B @ exp_A @ U
        
        diff = U - exacts[i]
        # Spectral norm is the largest singular value
        svals = np.linalg.svd(diff, compute_uv=False)
        total_error += svals[0]
        
    return total_error / len(t_values)

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
        # Nelder-Mead is robust for this small dimension
        # Use adaptive Nelder-Mead to handle potential flat spots
        res = minimize(
            _eval_error, 
            x0, 
            args=(A, B, eig_data_A, eig_data_B, exacts, t_values),
            method='Nelder-Mead',
            options={
                'maxiter': 2000, # Reduced to ensure we finish in time
                'xatol': 1e-7,
                'fatol': 1e-9,
                'adaptive': True
            }
        )
        
        if res.fun < best_err:
            best_err = res.fun
            best_params = res.x
            
        # Check time for a second refinement run
        elapsed = time.time() - start_time
        if elapsed < _DEADLINE - 15:
            res2 = minimize(
                _eval_error, 
                best_params, 
                args=(A, B, eig_data_A, eig_data_B, exacts, t_values),
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-9,
                    'fatol': 1e-11
                }
            )
            if res2.fun < best_err:
                best_err = res2.fun
                best_params = res2.x
                
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
