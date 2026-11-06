"""Product formula coefficient optimization for Heisenberg chain."""
import numpy as np
from scipy import linalg
import time

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    sx = 0.5 * np.array([[0, 1], [1, 0]])
    sy = 0.5 * np.array([[0, -1j], [1j, 0]])
    sz = 0.5 * np.array([[1, 0], [0, -1]])
    
    n = 6
    dim = 2 ** n
    delta = 0.8
    h = 0.3
    
    def site_op(op, site):
        result = np.array([[1.0]])
        for i in range(n):
            if i == site:
                result = np.kron(result, op)
            else:
                result = np.kron(result, np.eye(2))
        return result
    
    def bond_op(op1, op2, site1, site2):
        result = np.array([[1.0]])
        for i in range(n):
            if i == site1:
                result = np.kron(result, op1)
            elif i == site2:
                result = np.kron(result, op2)
            else:
                result = np.kron(result, np.eye(2))
        return result
    
    A = np.zeros((dim, dim), dtype=complex)
    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        A += bond_op(sx, sx, i, j) + bond_op(sy, sy, i, j) + delta * bond_op(sz, sz, i, j)
    for i in range(n):
        A += 0.5 * h * site_op(sz, i)
    
    B = np.zeros((dim, dim), dtype=complex)
    for (i, j) in [(1, 2), (3, 4)]:
        B += bond_op(sx, sx, i, j) + bond_op(sy, sy, i, j) + delta * bond_op(sz, sz, i, j)
    for i in range(n):
        B += 0.5 * h * site_op(sz, i)
    
    return A, B

def _compute_error_fast(a, b, A, B, t):
    """Compute spectral norm error at a single time."""
    dim = A.shape[0]
    H = A + B
    exact = linalg.expm(-1j * t * H)
    U = np.eye(dim, dtype=complex)
    for k in range(len(a)):
        Fk = linalg.expm(-1j * b[k] * t * B) @ linalg.expm(-1j * a[k] * t * A)
        U = Fk @ U
    return np.linalg.norm(U - exact, 2)

def coefficients(stages):
    """Find optimal coefficients for the 12-stage product formula."""
    start_time = time.time()
    deadline = start_time + 140
    
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    half = stages // 2
    
    # Initial guess: symmetric uniform
    a_half = [1.0 / stages] * half
    b_half = [1.0 / stages] * half
    
    def _make_full(a_h, b_h):
        a_full = a_h + a_h[::-1]
        b_full = b_h + b_h[::-1]
        return a_full, b_full
    
    def _err(a_h, b_h):
        a_full, b_full = _make_full(a_h, b_h)
        sa, sb = sum(a_full), sum(b_full)
        if abs(sa) > 1e-15:
            a_full = [x / sa for x in a_full]
        if abs(sb) > 1e-15:
            b_full = [x / sb for x in b_full]
        errs = [_compute_error_fast(a_full, b_full, A, B, t) for t in times]
        return sum(errs) / len(errs)
    
    best_a_h = a_half[:]
    best_b_h = b_half[:]
    best_err = _err(best_a_h, best_b_h)
    
    # Local search with random perturbations
    from scipy.optimize import minimize
    
    def _obj(x):
        a_h = list(x[:half])
        b_h = list(x[half:2*half])
        return _err(a_h, b_h)
    
    x0 = np.array(a_half + b_half)
    
    for attempt in range(8):
        if time.time() > deadline - 30:
            break
        
        if attempt == 0:
            x_start = x0
        else:
            x_start = x0 + 0.05 * np.random.randn(2 * half)
        
        try:
            res = minimize(_obj, x_start, method='Nelder-Mead',
                         options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_err:
                best_err = res.fun
                best_a_h = list(res.x[:half])
                best_b_h = list(res.x[half:2*half])
        except:
            pass
    
    a_full, b_full = _make_full(best_a_h, best_b_h)
    sa, sb = sum(a_full), sum(b_full)
    if abs(sa) > 1e-15:
        a_full = [x / sa for x in a_full]
    if abs(sb) > 1e-15:
        b_full = [x / sb for x in b_full]
    
    a_full = [max(-50, min(50, x)) for x in a_full]
    b_full = [max(-50, min(50, x)) for x in b_full]
    
    sa, sb = sum(a_full), sum(b_full)
    if abs(sa) > 1e-15:
        a_full = [x / sa for x in a_full]
    if abs(sb) > 1e-15:
        b_full = [x / sb for x in b_full]
    
    return a_full, b_full
