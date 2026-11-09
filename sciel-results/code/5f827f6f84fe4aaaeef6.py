"""Product-formula coefficient optimizer for 6-site Heisenberg chain."""
import numpy as np
from scipy import linalg
from scipy.optimize import minimize
import time

_T0 = time.time()
_DEADLINE = 130.0

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _build_hamiltonians():
    N = 6
    delta = 0.8
    field = 0.3
    
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    ident = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1.0]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    def bond_ham(i):
        ops = [ident] * N
        ops[i] = sx
        ops[i+1] = sx
        h1 = kron_all(ops)
        
        ops = [ident] * N
        ops[i] = sy
        ops[i+1] = sy
        h2 = kron_all(ops)
        
        ops = [ident] * N
        ops[i] = sz
        ops[i+1] = sz
        h3 = kron_all(ops)
        
        return h1 + h2 + delta * h3
    
    def field_term():
        h_field = np.zeros((2**N, 2**N), dtype=complex)
        for i in range(N):
            ops = [ident] * N
            ops[i] = sz
            h_field += kron_all(ops)
        return field * h_field
    
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    A += bond_ham(1)
    A += bond_ham(3)
    B += bond_ham(0)
    B += bond_ham(2)
    B += bond_ham(4)
    
    h_field = field_term()
    A += 0.5 * h_field
    B += 0.5 * h_field
    
    return A, B

def _objective(coeffs, A, B, times):
    a = coeffs[:12]
    b = coeffs[12:]
    
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    penalty = 1000.0 * ((sum_a - 1.0)**2 + (sum_b - 1.0)**2)
    
    total_err = 0.0
    dim = A.shape[0]
    I = np.eye(dim, dtype=complex)
    
    for t in times:
        U = I.copy()
        for k in range(12):
            if abs(b[k]) > 1e-15:
                U = linalg.expm(-1j * b[k] * t * B) @ U
            if abs(a[k]) > 1e-15:
                U = linalg.expm(-1j * a[k] * t * A) @ U
        
        U_exact = linalg.expm(-1j * t * (A + B))
        err = np.linalg.norm(U - U_exact, 2)
        total_err += err
    
    return total_err / len(times) + penalty

def coefficients(stages):
    global _T0
    _T0 = time.time()
    
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    x0 = np.array([1.0/12]*24)
    best_x = x0.copy()
    best_val = _objective(x0, A, B, times)
    
    np.random.seed(42)
    
    for r in range(10):
        if _remaining() < 25.0:
            break
        
        if r == 0:
            x_start = x0
        else:
            x_start = best_x + 0.05 * np.random.randn(24)
            x_start = np.clip(x_start, -50, 50)
        
        try:
            result = minimize(
                _objective,
                x_start,
                args=(A, B, times),
                method='L-BFGS-B',
                bounds=[(-50, 50)] * 24,
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            continue
    
    a = best_x[:12].copy()
    b = best_x[12:].copy()
    
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-10:
        a = a / sa
    if abs(sb) > 1e-10:
        b = b / sb
    
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-10:
        a = a / sa
    if abs(sb) > 1e-10:
        b = b / sb
    
    return list(a), list(b)
