"""Optimized 12-stage product formula for Heisenberg-chain time evolution."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm

def _build_hamiltonians():
    """Build A and B for 6-site XXZ Heisenberg chain with anisotropy 0.8, field 0.3."""
    N = 6
    dim = 2 ** N
    delta = 0.8  # anisotropy
    h = 0.3      # field
    
    # Spin operators for a single site (spin-1/2)
    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    sp = sx + 1j * sy  # raising: |0><1| (maps |down> to |up>)
    sm = sx - 1j * sy  # lowering: |1><0| (maps |up> to |down>)
    
    def kron_all(ops):
        """Kron product of all operators."""
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result
    
    def site_op(op, site):
        """Operator acting on a specific site."""
        ops = [I2] * N
        ops[site] = op
        return kron_all(ops)
    
    def bond_op(j):
        """Bond interaction between sites j and j+1 (0-indexed).
        H_bond = 0.5*(S_j^+ S_{j+1}^- + S_j^- S_{j+1}^+) + delta * S_j^z S_{j+1}^z
        """
        # S_j^+ S_{j+1}^-
        ops = [I2] * N
        ops[j] = sp
        ops[j+1] = sm
        term1 = kron_all(ops)
        
        # S_j^- S_{j+1}^+
        ops = [I2] * N
        ops[j] = sm
        ops[j+1] = sp
        term2 = kron_all(ops)
        
        # S_j^z S_{j+1}^z
        ops = [I2] * N
        ops[j] = sz
        ops[j+1] = sz
        term3 = kron_all(ops)
        
        return 0.5 * (term1 + term2) + delta * term3
    
    # Total field: h * sum_j S_j^z
    field = np.zeros((dim, dim), dtype=complex)
    for j in range(N):
        field += site_op(sz, j)
    
    # A: odd bonds (0-1, 2-3, 4-5) + half field
    A = np.zeros((dim, dim), dtype=complex)
    for j in [0, 2, 4]:
        A += bond_op(j)
    A += 0.5 * h * field
    
    # B: even bonds (1-2, 3-4) + half field
    B = np.zeros((dim, dim), dtype=complex)
    for j in [1, 3]:
        B += bond_op(j)
    B += 0.5 * h * field
    
    return A, B

def _spectral_norm_error(a, b, A, B, times):
    """Compute mean spectral norm error over given times."""
    total_err = 0.0
    n = len(times)
    for t in times:
        U = np.eye(A.shape[0], dtype=complex)
        for k in range(len(a)):
            U = expm(-1j * b[k] * t * B) @ expm(-1j * a[k] * t * A) @ U
        U_exact = expm(-1j * t * (A + B))
        err = np.linalg.norm(U - U_exact, 2)
        total_err += err
    return total_err / n

def coefficients(stages):
    """Find optimal 12-stage product formula coefficients."""
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    if stages != 12:
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    A, B = _build_hamiltonians()
    AB = A + B
    times = [0.25, 0.5, 1.0]
    
    # Precompute expm for exact propagator
    U_exact = {}
    for t in times:
        U_exact[t] = expm(-1j * t * AB)
    
    def error_func(a, b):
        total_err = 0.0
        n = len(times)
        for t in times:
            U = np.eye(64, dtype=complex)
            for k in range(12):
                U = expm(-1j * b[k] * t * B) @ expm(-1j * a[k] * t * A) @ U
            err = np.linalg.norm(U - U_exact[t], 2)
            total_err += err
        return total_err / n
    
    # Constrained optimization: 22 free params (11 for a, 11 for b)
    def objective(x):
        a = list(x[:11])
        b = list(x[11:22])
        a.append(1.0 - sum(a))
        b.append(1.0 - sum(b))
        # Clamp
        a = [max(-50, min(50, ai)) for ai in a]
        b = [max(-50, min(50, bi)) for bi in b]
        # Normalize
        a_sum = sum(a)
        b_sum = sum(b)
        if abs(a_sum) < 1e-10 or abs(b_sum) < 1e-10:
            return 1e10
        a = [ai / a_sum for ai in a]
        b = [bi / b_sum for bi in b]
        return error_func(a, b)
    
    # Initial guess
    x0 = [1.0/12] * 22
    best_x = x0.copy()
    best_f = objective(x0)
    
    # Nelder-Mead
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10,
                                  'adaptive': True})
        if result.fun < best_f:
            best_f = result.fun
            best_x = result.x
    except Exception:
        pass
    
    if time.time() > deadline - 10.0:
        a = list(best_x[:11])
        b = list(best_x[11:22])
        a.append(1.0 - sum(a))
        b.append(1.0 - sum(b))
        a = [max(-50, min(50, ai)) for ai in a]
        b = [max(-50, min(50, bi)) for bi in b]
        a_sum = sum(a)
        b_sum = sum(b)
        if abs(a_sum) > 1e-10:
            a = [ai / a_sum for ai in a]
        if abs(b_sum) > 1e-10:
            b = [bi / b_sum for bi in b]
        return a, b
    
    # L-BFGS-B refinement
    try:
        result2 = minimize(objective, best_x, method='L-BFGS-B',
                          options={'maxiter': 200, 'gtol': 1e-8})
        if result2.fun < best_f:
            best_f = result2.fun
            best_x = result2.x
    except Exception:
        pass
    
    # Extract
    a = list(best_x[:11])
    b = list(best_x[11:22])
    a.append(1.0 - sum(a))
    b.append(1.0 - sum(b))
    a = [max(-50, min(50, ai)) for ai in a]
    b = [max(-50, min(50, bi)) for bi in b]
    a_sum = sum(a)
    b_sum = sum(b)
    if abs(a_sum) > 1e-10:
        a = [ai / a_sum for ai in a]
    if abs(b_sum) > 1e-10:
        b = [bi / b_sum for bi in b]
    
    assert len(a) == 12 and len(b) == 12
    assert all(abs(ai) <= 50 for ai in a)
    assert all(abs(bi) <= 50 for bi in b)
    assert abs(sum(a) - 1.0) < 1e-6
    assert abs(sum(b) - 1.0) < 1e-6
    
    return a, b
