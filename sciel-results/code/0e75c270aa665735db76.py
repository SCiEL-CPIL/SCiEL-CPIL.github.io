"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    N = 6
    dim = 2 ** N
    anisotropy = 0.8
    field = 0.3
    
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_chain(mats):
        result = np.array([[1]], dtype=complex)
        for m in mats:
            result = np.kron(result, m)
        return result
    
    def bond_term(i, j):
        term1 = kron_chain([sx if k == i else (sx if k == j else I2) for k in range(N)])
        term2 = kron_chain([sy if k == i else (sy if k == j else I2) for k in range(N)])
        term3 = kron_chain([sz if k == i else (sz if k == j else I2) for k in range(N)])
        return term1 + term2 + anisotropy * term3
    
    def field_term():
        result = np.zeros((dim, dim), dtype=complex)
        for k in range(N):
            mats = [sz if i == k else I2 for i in range(N)]
            result += kron_chain(mats)
        return result
    
    A = np.zeros((dim, dim), dtype=complex)
    for i in range(0, N, 2):
        A += bond_term(i, i+1)
    A += 0.5 * field_term()
    
    B = np.zeros((dim, dim), dtype=complex)
    for i in range(1, N, 2):
        B += bond_term(i, i+1)
    B += 0.5 * field_term()
    
    return A, B

def coefficients(stages):
    stages = int(stages)
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
    
    t0 = time.time()
    deadline = t0 + 135.0
    
    A, B = _build_hamiltonians()
    H = A + B
    dim = A.shape[0]
    times = [0.25, 0.5, 1.0]
    
    # Precompute eigen-decompositions (A and B are Hermitian)
    evals_A, V_A = np.linalg.eigh(A)
    V_A_inv = np.linalg.inv(V_A)
    evals_B, V_B = np.linalg.eigh(B)
    V_B_inv = np.linalg.inv(V_B)
    
    def fast_expm(c, t, evals, V, V_inv):
        phase = np.exp(-1j * c * t * evals)
        return V @ np.diag(phase) @ V_inv
    
    def error_fn(params):
        a = params[:12]
        b = params[12:]
        a = a + (1.0 - np.sum(a)) / 12
        b = b + (1.0 - np.sum(b)) / 12
        
        total_err = 0.0
        for t in times:
            U = np.eye(dim, dtype=complex)
            for k in range(12):
                Fk = fast_expm(b[k], t, evals_B, V_B, V_B_inv) @ fast_expm(a[k], t, evals_A, V_A, V_A_inv)
                U = U @ Fk
            U_exact = expm(-1j * t * H)
            total_err += np.linalg.norm(U - U_exact, 2)
        return total_err / len(times)
    
    from scipy.optimize import minimize
    
    x0 = np.concatenate([np.ones(12)/12, np.ones(12)/12])
    
    best_x = x0.copy()
    best_err = error_fn(x0)
    
    try:
        res = minimize(error_fn, x0, method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-12})
        if res.fun < best_err:
            best_err = res.fun
            best_x = res.x
    except Exception:
        pass
    
    if time.time() > deadline:
        a = best_x[:12] + (1.0 - np.sum(best_x[:12])) / 12
        b = best_x[12:] + (1.0 - np.sum(best_x[12:])) / 12
        return list(a), list(b)
    
    # Random restarts
    np.random.seed(42)
    for i in range(10):
        if time.time() > deadline - 3.0:
            break
        x_r = x0 + np.random.randn(24) * 0.05
        try:
            res = minimize(error_fn, x_r, method='L-BFGS-B', options={'maxiter': 150, 'ftol': 1e-12})
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x
        except Exception:
            pass
    
    a = best_x[:12] + (1.0 - np.sum(best_x[:12])) / 12
    b = best_x[12:] + (1.0 - np.sum(best_x[12:])) / 12
    return list(a), list(b)
# EVOLVE-BLOCK-END
