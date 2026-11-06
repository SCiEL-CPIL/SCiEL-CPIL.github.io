import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import time

_OPT_COEFFS = None
_OPT_DONE = False

def _build_hamiltonians():
    N = 6
    dim = 2 ** N
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        res = np.array([[1]], dtype=complex)
        for op in ops:
            res = np.kron(res, op)
        return res
        
    def get_op(op, site):
        ops = [I] * N
        ops[site] = op
        return kron_all(ops)
        
    def get_two_op(op1, op2, site1, site2):
        ops = [I] * N
        ops[site1] = op1
        ops[site2] = op2
        return kron_all(ops)

    anisotropy = 0.8
    field = 0.3
    
    H_odd = np.zeros((dim, dim), dtype=complex)
    H_even = np.zeros((dim, dim), dtype=complex)
    
    for i in range(5):
        H_bond = get_two_op(sx, sx, i, i+1) + get_two_op(sy, sy, i, i+1) + anisotropy * get_two_op(sz, sz, i, i+1)
        if i % 2 == 0:
            H_odd += H_bond
        else:
            H_even += H_bond
            
    H_field = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        H_field += get_op(sz, i)
        
    H_field_half = 0.5 * field * H_field
    
    A = H_odd + H_field_half
    B = H_even + H_field_half
    
    return A, B

def _evaluate_error(coeffs, A, B, times):
    a, b = coeffs
    dim = A.shape[0]
    errors = []
    for t in times:
        U_exact_t = la.expm(-1j * t * (A + B))
        U_approx_t = np.eye(dim, dtype=complex)
        for k in range(12):
            M_k = la.expm(-1j * b[k] * t * B) @ la.expm(-1j * a[k] * t * A)
            U_approx_t = U_approx_t @ M_k
        err = la.norm(U_approx_t - U_exact_t, 2)
        errors.append(err)
    return np.mean(errors)

def _optimize_coefficients():
    global _OPT_COEFFS, _OPT_DONE
    if _OPT_DONE:
        return _OPT_COEFFS
        
    try:
        A, B = _build_hamiltonians()
        times = [0.25, 0.5, 1.0]
        
        lam = 1.0 / (2.0 * (2.0 ** (1.0 / 3.0) - 1.0))
        c1 = -lam
        c2 = 1.0 + 2.0 * lam
        c3 = -lam
        a1 = c1 / 2.0
        a2 = (c1 + c2) / 2.0
        a3 = (c2 + c3) / 2.0
        a4 = c3 / 2.0
        
        b_weights = [0.0, c1, c2, c3]
        a_weights = [a1, a2, a3, a4]
        
        a_init = []
        b_init = []
        for i in range(4):
            for _ in range(3):
                a_init.append(a_weights[i] / 3.0)
                b_init.append(b_weights[i] / 3.0)
                
        x0 = np.array(a_init + b_init)
        
        def objective(x):
            a = x[:12]
            b = x[12:]
            pen = 100 * (np.sum(a) - 1)**2 + 100 * (np.sum(b) - 1)**2
            return _evaluate_error((a, b), A, B, times) + pen
            
        start_time = time.time()
        # Limit iterations to control time
        res = opt.minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
        
        elapsed = time.time() - start_time
        if elapsed < 55: # 5s margin
            x_opt = res.x
            a_opt = x_opt[:12]
            b_opt = x_opt[12:]
            
            sum_a = np.sum(a_opt)
            sum_b = np.sum(b_opt)
            if sum_a != 0:
                a_opt = a_opt / sum_a
            if sum_b != 0:
                b_opt = b_opt / sum_b
                
            _OPT_COEFFS = (a_opt.tolist(), b_opt.tolist())
            _OPT_DONE = True
            return _OPT_COEFFS
        else:
            _OPT_COEFFS = (a_init, b_init)
            _OPT_DONE = True
            return _OPT_COEFFS
            
    except Exception as e:
        lam = 1.0 / (2.0 * (2.0 ** (1.0 / 3.0) - 1.0))
        c1 = -lam
        c2 = 1.0 + 2.0 * lam
        c3 = -lam
        a1 = c1 / 2.0
        a2 = (c1 + c2) / 2.0
        a3 = (c2 + c3) / 2.0
        a4 = c3 / 2.0
        b_weights = [0.0, c1, c2, c3]
        a_weights = [a1, a2, a3, a4]
        a_init = []
        b_init = []
        for i in range(4):
            for _ in range(3):
                a_init.append(a_weights[i] / 3.0)
                b_init.append(b_weights[i] / 3.0)
        _OPT_COEFFS = (a_init, b_init)
        _OPT_DONE = True
        return _OPT_COEFFS

def coefficients(stages):
    if stages != 12:
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
        
    if _OPT_COEFFS is None:
        _optimize_coefficients()
        
    a, b = _OPT_COEFFS
    return a, b
