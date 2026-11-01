"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import least_squares

def _verify_factors(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W with shapes (n*m, rank), (m*p, rank), (p*n, rank)
    produce the matrix multiplication tensor under the trace convention W[k*n+i, r]."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                actual = np.sum(U[A, :] * V[B, :] * W[C, :])
                if abs(actual - 1) > tol:
                    return False, (A, B, C, 1, actual)
    
    # Check invalid entries
    for A in range(n * m):
        for B in range(m * p):
            for C in range(p * n):
                is_valid = False
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if A == i * m + j and B == j * p + k and C == k * n + i:
                                is_valid = True
                if not is_valid:
                    actual = np.sum(U[A, :] * V[B, :] * W[C, :])
                    if abs(actual) > tol:
                        return False, (A, B, C, 0, actual)
    
    return True, None

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _solve_for_rank(n, m, p, rank, seed=42):
    """Try to find a rank-rank decomposition using optimization."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build all triples with expected values
    all_triples = []
    for A in range(nm):
        for B in range(mp):
            for C in range(pn):
                expected = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if A == i * m + j and B == j * p + k and C == k * n + i:
                                expected = 1
                all_triples.append((A, B, C, expected))
    
    def residuals(x):
        U = x[:nm*rank].reshape(nm, rank)
        V = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W = x[nm*rank+mp*rank:].reshape(pn, rank)
        
        res = []
        for A, B, C, expected in all_triples:
            actual = np.sum(U[A, :] * V[B, :] * W[C, :])
            res.append(actual - expected)
        return res
    
    np_rng = np.random.RandomState(seed)
    
    for attempt in range(30):
        # Initialize
        if attempt == 0:
            # Schoolbook-like initialization
            x0 = np.zeros(nm*rank + mp*rank + pn*rank)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        if idx < rank:
                            A = i * m + j
                            B = j * p + k
                            C = k * n + i
                            x0[A*rank + idx] = 1
                            x0[nm*rank + B*rank + idx] = 1
                            x0[nm*rank + mp*rank + C*rank + idx] = 1
        else:
            x0 = np_rng.normal(0, 1, size=(nm*rank + mp*rank + pn*rank))
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=300)
            if np.max(np.abs(result.fun)) < 1e-6:
                U = result.x[:nm*rank].reshape(nm, rank)
                V = result.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                W = result.x[nm*rank+mp*rank:].reshape(pn, rank)
                
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                ok, _ = _verify_factors(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p)
                if ok:
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
                
                ok, _ = _verify_factors(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if ok:
                    return U.tolist(), V.tolist(), W.tolist()
        except Exception:
            continue
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = 120 - 15
    start_time = time.time()
    
    # Try to find a low-rank decomposition
    for target_rank in [11, 12, 13, 14, 15, 16, 17]:
        if time.time() - start_time > deadline:
            break
        
        result = _solve_for_rank(n, m, p, target_rank)
        if result is not None:
            U, V, W = result
            ok, _ = _verify_factors(U, V, W, n, m, p)
            if ok:
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook (rank 18)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
