import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 130.0
    
    n_sites = 6
    dim = 2 ** n_sites
    
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
    ident = np.eye(2, dtype=complex)
    
    def site_op(op, i):
        ops = [ident] * n_sites
        ops[i] = op
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result
    
    anisotropy = 0.8
    field = 0.3
    J = 1.0
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    def bond_term(i, j):
        sxi = site_op(sx, i)
        sxj = site_op(sx, j)
        syi = site_op(sy, i)
        syj = site_op(sy, j)
        szi = site_op(sz, i)
        szj = site_op(sz, j)
        return J * (sxi @ sxj + syi @ syj + anisotropy * szi @ szj)
    
    for i, j in [(0, 1), (2, 3), (4, 5)]:
        A += bond_term(i, j)
    for i, j in [(1, 2), (3, 4)]:
        B += bond_term(i, j)
    
    field_total = np.zeros((dim, dim), dtype=complex)
    for i in range(n_sites):
        field_total += site_op(sz, i)
    
    A += (field / 2) * field_total
    B += (field / 2) * field_total
    
    H = A + B
    
    t_values = [0.25, 0.5, 1.0]
    exact_U = [expm(-1j * t * H) for t in t_values]
    
    # Precompute eigendecompositions
    A_eigvals, A_eigvecs = np.linalg.eig(A)
    B_eigvals, B_eigvecs = np.linalg.eig(B)
    A_inv_eigvecs = np.linalg.inv(A_eigvecs)
    B_inv_eigvecs = np.linalg.inv(B_eigvecs)
    
    eye = np.eye(dim, dtype=complex)
    
    def fast_expm_A(c):
        if abs(c) < 1e-15:
            return eye.copy()
        phase = np.exp(-1j * c * A_eigvals)
        return (A_eigvecs * phase) @ A_inv_eigvecs
    
    def fast_expm_B(c):
        if abs(c) < 1e-15:
            return eye.copy()
        phase = np.exp(-1j * c * B_eigvals)
        return (B_eigvecs * phase) @ B_inv_eigvecs
    
    def compute_error(a, b):
        total_err = 0.0
        for idx, t in enumerate(t_values):
            U = eye.copy()
            for k in range(12):
                if abs(b[k]) > 1e-15:
                    U = fast_expm_B(b[k] * t) @ U
                if abs(a[k]) > 1e-15:
                    U = fast_expm_A(a[k] * t) @ U
            err = np.linalg.norm(exact_U[idx] - U, 2)
            total_err += err
        return total_err / 3.0
    
    # Symmetric parameterization: a = [a1..a6, a6..a1], b = [b1..b6, b6..b1]
    def objective_sym(params):
        a_half = params[:6]
        b_half = params[6:12]
        a = np.array([a_half[0], a_half[1], a_half[2], a_half[3], a_half[4], a_half[5],
                     a_half[5], a_half[4], a_half[3], a_half[2], a_half[1], a_half[0]])
        b = np.array([b_half[0], b_half[1], b_half[2], b_half[3], b_half[4], b_half[5],
                     b_half[5], b_half[4], b_half[3], b_half[2], b_half[1], b_half[0]])
        return compute_error(a, b)
    
    # Initial guess: uniform
    x0 = np.ones(12) / 12
    constraints_sym = [
        {'type': 'eq', 'fun': lambda x: 2 * np.sum(x[:6]) - 1.0},
        {'type': 'eq', 'fun': lambda x: 2 * np.sum(x[6:12]) - 1.0}
    ]
    bounds_sym = [(-50, 50)] * 12
    
    best_x = x0.copy()
    best_obj = objective_sym(x0)
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        try:
            result = minimize(objective_sym, x0, method='SLSQP', bounds=bounds_sym,
                             constraints=constraints_sym, options={'maxiter': 300, 'ftol': 1e-14})
            if result.fun < best_obj:
                best_x = result.x
                best_obj = result.fun
        except Exception:
            pass
    
    # Try a few random restarts if time permits
    elapsed = time.time() - start_time
    rng = np.random.RandomState(42)
    for _ in range(5):
        if time.time() - start_time > deadline - 10:
            break
        try:
            x0_rand = rng.uniform(-1, 1, 12)
            # Normalize to satisfy constraints
            a_half = x0_rand[:6]
            b_half = x0_rand[6:12]
            sa = 2 * np.sum(a_half)
            sb = 2 * np.sum(b_half)
            if abs(sa) > 1e-10:
                a_half = a_half / sa
            if abs(sb) > 1e-10:
                b_half = b_half / sb
            x0_rand = np.concatenate([a_half, b_half])
            
            result_r = minimize(objective_sym, x0_rand, method='SLSQP', bounds=bounds_sym,
                               constraints=constraints_sym, options={'maxiter': 100, 'ftol': 1e-12})
            if result_r.fun < best_obj:
                best_x = result_r.x
                best_obj = result_r.fun
        except Exception:
            pass
    
    # Convert to full 24 coefficients
    a_half = best_x[:6]
    b_half = best_x[6:12]
    a_full = np.array([a_half[0], a_half[1], a_half[2], a_half[3], a_half[4], a_half[5],
                      a_half[5], a_half[4], a_half[3], a_half[2], a_half[1], a_half[0]])
    b_full = np.array([b_half[0], b_half[1], b_half[2], b_half[3], b_half[4], b_half[5],
                      b_half[5], b_half[4], b_half[3], b_half[2], b_half[1], b_half[0]])
    
    a_list = [float(max(-50, min(50, x))) for x in a_full]
    b_list = [float(max(-50, min(50, x))) for x in b_full]
    
    sa = sum(a_list)
    sb = sum(b_list)
    if abs(sa) > 1e-15:
        a_list = [x / sa for x in a_list]
    if abs(sb) > 1e-15:
        b_list = [x / sb for x in b_list]
    
    return a_list, b_list
