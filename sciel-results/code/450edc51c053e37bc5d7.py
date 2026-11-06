import numpy as np
import time
import scipy.linalg as la

def _build_heisenberg_matrices():
    """Build A and B matrices for 6-site XXZ Heisenberg chain."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    anisotropy = 0.8
    field = 0.3
    n_sites = 6
    dim = 2 ** n_sites
    
    def op_on_site(op, site):
        ops = [I2] * n_sites
        ops[site] = op
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def bond_term(i, j):
        sx_i = op_on_site(sx, i)
        sx_j = op_on_site(sx, j)
        sy_i = op_on_site(sy, i)
        sy_j = op_on_site(sy, j)
        sz_i = op_on_site(sz, i)
        sz_j = op_on_site(sz, j)
        return 0.5 * (sx_i @ sx_j + sy_i @ sy_j) + anisotropy * (sz_i @ sz_j)
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for i in [0, 2, 4]:
        A += bond_term(i, i + 1)
    for i in [1, 3]:
        B += bond_term(i, i + 1)
    
    for i in range(n_sites):
        half_field = (field / 2) * op_on_site(sz, i)
        A += half_field
        B += half_field
    
    return A, B

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 140
    
    A, B = _build_heisenberg_matrices()
    dim = A.shape[0]
    t_values = np.array([0.25, 0.5, 1.0])
    
    # Precompute expm for various coefficients to speed up
    # Use 4th-order Suzuki-Trotter as initial guess
    z = 1.0 / (2.0 - 2.0 ** (1.0/3.0))
    
    # 4th-order symmetric: A(z) B(1-z) A(z) B(z) A(1-z) B(z)
    # Map to our format: need to express as (B_k, A_k) pairs
    # This is tricky. Let's just use uniform as start and optimize.
    
    a = np.ones(stages) / stages
    b = np.ones(stages) / stages
    
    def error_fn(x):
        a = x[:stages]
        b = x[stages:]
        total_err = 0.0
        for t in t_values:
            U = np.eye(dim, dtype=complex)
            for k in range(stages):
                U = la.expm(-1j * b[k] * t * B) @ U
                U = la.expm(-1j * a[k] * t * A) @ U
            H = A + B
            U_exact = la.expm(-1j * t * H)
            diff = U - U_exact
            s = np.linalg.svd(diff, compute_uv=False)
            total_err += s[0] if len(s) > 0 else 0.0
        return total_err / len(t_values)
    
    x0 = np.concatenate([a, b])
    best_x = x0.copy()
    best_val = error_fn(x0)
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        from scipy.optimize import minimize
        try:
            res = minimize(error_fn, x0, method='Nelder-Mead', 
                         options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-4})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    a_opt = list(best_x[:stages])
    b_opt = list(best_x[stages:])
    
    # Clamp and normalize
    a_opt = [max(-50, min(50, float(v))) if np.isfinite(v) else 0.0 for v in a_opt]
    b_opt = [max(-50, min(50, float(v))) if np.isfinite(v) else 0.0 for v in b_opt]
    
    sa = sum(a_opt)
    sb = sum(b_opt)
    if abs(sa) > 1e-12:
        a_opt = [v / sa for v in a_opt]
    else:
        a_opt = [1.0/stages]*stages
    if abs(sb) > 1e-12:
        b_opt = [v / sb for v in b_opt]
    else:
        b_opt = [1.0/stages]*stages
    
    return a_opt, b_opt
