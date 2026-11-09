import numpy as np
import time
from scipy.optimize import minimize

# Global cache for Hamiltonians and their decompositions
_HAM_CACHE = {}

def _build_hamiltonians():
    """
    Build A (odd bonds + half field) and B (even bonds + half field) 
    for a 6-site open XXZ Heisenberg chain.
    """
    if 'A' in _HAM_CACHE:
        return _HAM_CACHE['A'], _HAM_CACHE['B']
        
    N = 6
    delta = 0.8
    field = 0.3
    
    # Spin-1/2 operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1.0+0j]])
        for op in ops:
            result = np.kron(result, op)
        return result

    def site_op(op, site):
        ops = [I2] * N
        ops[site] = op
        return kron_all(ops)

    # Build A: Odd bonds (0,1), (2,3), (4,5) + half field
    A = np.zeros((2**N, 2**N), dtype=complex)
    # Bonds
    for i in range(0, N, 2):
        j = i + 1
        # Sx_i Sx_j + Sy_i Sy_j + delta Sz_i Sz_j
        term_xx = site_op(sx, i) @ site_op(sx, j)
        term_yy = site_op(sy, i) @ site_op(sy, j)
        term_zz = site_op(sz, i) @ site_op(sz, j)
        A += term_xx + term_yy + delta * term_zz
        
    # Field: -h/2 sum Sz_i
    for i in range(N):
        A -= (field / 2.0) * site_op(sz, i)

    # Build B: Even bonds (1,2), (3,4) + half field
    B = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(1, N-1, 2):
        j = i + 1
        term_xx = site_op(sx, i) @ site_op(sx, j)
        term_yy = site_op(sy, i) @ site_op(sy, j)
        term_zz = site_op(sz, i) @ site_op(sz, j)
        B += term_xx + term_yy + delta * term_zz
        
    for i in range(N):
        B -= (field / 2.0) * site_op(sz, i)

    _HAM_CACHE['A'] = A
    _HAM_CACHE['B'] = B
    return A, B

def coefficients(stages):
    """
    Optimizes coefficients for a 12-stage product formula.
    Returns (a, b) lists of length 12.
    """
    if stages != 12:
        # Fallback for non-12 stages
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill

    A, B = _build_hamiltonians()
    
    # Pre-compute eigen-decompositions for fast exponentiation
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    
    H_total = A + B
    eigvals_H, eigvecs_H = np.linalg.eigh(H_total)
    
    def expm_hermitian(theta, eigvals, eigvecs):
        """Compute exp(-i * theta * H) using precomputed eigh."""
        phases = np.exp(-1j * theta * eigvals)
        V_phases = eigvecs * phases
        return V_phases @ eigvecs.conj().T

    def evaluate_error(params):
        """
        params: array of 22 floats (a1..a11, b1..b11)
        Returns mean spectral norm error.
        """
        a = np.zeros(12)
        b = np.zeros(12)
        a[:11] = params[:11]
        b[:11] = params[11:]
        
        a[11] = 1.0 - np.sum(a[:11])
        b[11] = 1.0 - np.sum(b[:11])
        
        # Check bounds
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e10
            
        t_values = [0.25, 0.5, 1.0]
        errors = []
        
        for t in t_values:
            # Compute Product Formula U(t)
            U = np.eye(2**6, dtype=complex)
            
            # Order: k=1..12, exp(-i b_k t B) exp(-i a_k t A)
            for k in range(12):
                # Exp B
                theta_B = b[k] * t
                if abs(theta_B) > 1e-12:
                    E_B = expm_hermitian(theta_B, eigvals_B, eigvecs_B)
                    U = U @ E_B
                # Exp A
                theta_A = a[k] * t
                if abs(theta_A) > 1e-12:
                    E_A = expm_hermitian(theta_A, eigvals_A, eigvecs_A)
                    U = U @ E_A
            
            # Exact Propagator
            E_exact = expm_hermitian(t, eigvals_H, eigvecs_H)
            
            # Error
            diff = U - E_exact
            try:
                err = np.linalg.norm(diff, 2)
            except:
                err = 1e10
            errors.append(err)
            
        return np.mean(errors)

    # Initial guess: Strang splitting (uniform)
    x0 = np.full(22, 1.0/12.0)
    
    # Optimization
    best_x = x0.copy()
    best_err = evaluate_error(x0)
    
    # Use Nelder-Mead
    iterations = 0
    max_iter = 2000
    
    # Callback to check time
    def callback(xk):
        nonlocal best_x, best_err
        # Only evaluate if we have time
        if time.time() - start_time > time_limit:
            raise Exception("Time limit reached")
        
        current_err = evaluate_error(xk)
        if current_err < best_err:
            best_err = current_err
            best_x = xk.copy()
            
        return current_err

    try:
        res = minimize(
            evaluate_error, 
            x0, 
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6},
            callback=callback
        )
        
        # Update best if the final result is better
        if time.time() - start_time < time_limit:
            final_err = evaluate_error(res.x)
            if final_err < best_err:
                best_err = final_err
                best_x = res.x.copy()
                
    except Exception as e:
        # If optimization fails or times out, use best_x
        pass

    # Construct final a and b
    a_opt = np.zeros(12)
    b_opt = np.zeros(12)
    a_opt[:11] = best_x[:11]
    b_opt[:11] = best_x[11:]
    a_opt[11] = 1.0 - np.sum(a_opt[:11])
    b_opt[11] = 1.0 - np.sum(b_opt[:11])
    
    # Clip to valid range just in case
    a_opt = np.clip(a_opt, -50, 50)
    b_opt = np.clip(b_opt, -50, 50)
    
    # Re-normalize if clipping affected sum
    sum_a = np.sum(a_opt)
    sum_b = np.sum(b_opt)
    if abs(sum_a - 1.0) > 1e-9:
        a_opt = a_opt / sum_a
    if abs(sum_b - 1.0) > 1e-9:
        b_opt = b_opt / sum_b

    return list(a_opt), list(b_opt)
