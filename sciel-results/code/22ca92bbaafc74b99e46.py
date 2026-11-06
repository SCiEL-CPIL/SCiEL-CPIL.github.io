import numpy as np
import time
from scipy.optimize import minimize

def _build_hamiltonians():
    N = 6
    delta = 0.8
    h = 0.3
    
    I = np.eye(2, dtype=complex)
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    def kron_all(ops):
        out = np.array([[1.0]], dtype=complex)
        for op in ops:
            out = np.kron(out, op)
        return out
    
    A = np.zeros((64, 64), dtype=complex)
    B = np.zeros((64, 64), dtype=complex)
    
    H_field_half = 0.0
    for j in range(N):
        ops = [I] * N
        ops[j] = Sz
        H_field_half -= 0.5 * h * kron_all(ops)
        
    A += H_field_half
    B += H_field_half
    
    for j in [0, 2, 4]:
        ops = [I] * N
        ops[j] = Sx
        ops[j+1] = Sx
        A += kron_all(ops)
        
        ops = [I] * N
        ops[j] = Sy
        ops[j+1] = Sy
        A += kron_all(ops)
        
        ops = [I] * N
        ops[j] = Sz
        ops[j+1] = Sz
        A += delta * kron_all(ops)
        
    for j in [1, 3]:
        ops = [I] * N
        ops[j] = Sx
        ops[j+1] = Sx
        B += kron_all(ops)
        
        ops = [I] * N
        ops[j] = Sy
        ops[j+1] = Sy
        B += kron_all(ops)
        
        ops = [I] * N
        ops[j] = Sz
        ops[j+1] = Sz
        B += delta * kron_all(ops)
        
    return A, B

def _error_func(params, A, B, times):
    a_half = params[0:6]
    b_half = params[6:12]
    
    a = np.concatenate([a_half, a_half[::-1]])
    b = np.concatenate([b_half, b_half[::-1]])
    
    if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
        return 1e10
    
    err = 0.0
    for t in times:
        U = np.eye(64, dtype=complex)
        for k in range(12):
            U = np.linalg.multi_dot([
                np.linalg.expm(-1j * b[k] * t * B),
                np.linalg.expm(-1j * a[k] * t * A),
                U
            ])
            
        H = A + B
        U_exact = np.linalg.expm(-1j * t * H)
        
        diff = U - U_exact
        s = np.linalg.svd(diff, compute_uv=False)
        err += s[0]
        
    return err / len(times)

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 50.0
    
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    a_init = np.ones(6) * (1.0/12.0)
    b_init = np.ones(6) * (1.0/12.0)
    x0 = np.concatenate([a_init, b_init])
    
    best_x = x0
    best_err = _error_func(x0, A, B, times)
    
    try:
        res = minimize(_error_func, x0, args=(A, B, times), method='Powell', 
                       options={'maxiter': 200, 'maxfev': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if time.time() < deadline:
             if res.fun < best_err:
                 best_x = res.x
                 best_err = res.fun
    except Exception:
        pass
        
    a_half = best_x[0:6]
    b_half = best_x[6:12]
    a = np.concatenate([a_half, a_half[::-1]])
    b = np.concatenate([b_half, b_half[::-1]])
    
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    
    if sum_a != 0:
        a = a / sum_a
    if sum_b != 0:
        b = b / sum_b
        
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    return a.tolist(), b.tolist()
