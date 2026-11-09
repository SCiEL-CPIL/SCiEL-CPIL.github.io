import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 56
    deadline = time.time() + 140

    def verify(U, V, W, tol=1e-9):
        """Exact verification: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
        nm = n * m
        mp = m * p
        pn = p * n
        r = U.shape[1]
        Uf = np.asarray(U, dtype=float)
        Vf = np.asarray(V, dtype=float)
        Wf = np.asarray(W, dtype=float)
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    urow = Uf[i*m+j, :]
                    vrow = Vf[j*p+k, :]
                    wrow = Wf[k*n+i, :]
                    s += np.dot(urow, vrow * wrow)
                if abs(s - 1.0) > tol:
                    return False
        return True

    def snap_half(M):
        return np.round(np.asarray(M, dtype=float) * 2.0) / 2.0

    def to_list(M):
        r = M.shape[1]
        return [[float(M[i, j]) for j in range(r)] for i in range(M.shape[0])]

    # New structural approach: use bilinear algorithm construction with
    # systematic search over structured templates and rank reduction.
    
    def try_decomposition(r, rng):
        """Try to find a rank-r decomposition using structured search."""
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Strategy 1: Random search with structured initialization
        for attempt in range(200):
            if time.time() > deadline - 5:
                return None
            
            # Use a mix of random and structured patterns
            if attempt % 3 == 0:
                # Pure random with small values
                vals = np.array([-1, 0, 1], dtype=float)
                U = rng.choice(vals, size=(nm, r))
                W = rng.choice(vals, size=(pn, r))
            elif attempt % 3 == 1:
                # Structured: block diagonal patterns
                U = np.zeros((nm, r), dtype=float)
                W = np.zeros((pn, r), dtype=float)
                # Assign random values to certain blocks
                for c in range(r):
                    if rng.random() < 0.5:
                        U[rng.randint(nm), c] = rng.choice([-1, 1])
                    if rng.random() < 0.5:
                        W[rng.randint(pn), c] = rng.choice([-1, 1])
            else:
                # Sparse random
                U = np.zeros((nm, r), dtype=float)
                W = np.zeros((pn, r), dtype=float)
                for c in range(r):
                    nz_u = rng.randint(2, 6)
                    nz_w = rng.randint(2, 6)
                    rows_u = rng.choice(nm, size=nz_u, replace=False)
                    rows_w = rng.choice(pn, size=nz_w, replace=False)
                    U[rows_u, c] = rng.choice([-1, 1], size=nz_u)
                    W[rows_w, c] = rng.choice([-1, 1], size=nz_w)
            
            V = np.zeros((mp, r), dtype=float)
            valid = True
            
            for j in range(m):
                for k in range(p):
                    B = np.array([
                        [U[i*m+j, c] * W[k*n+i, c] for c in range(r)]
                        for i in range(n)
                    ], dtype=float)
                    rhs = np.ones(n)
                    try:
                        v = np.linalg.lstsq(B, rhs, rcond=None)[0]
                        if not np.allclose(B @ v, rhs, atol=1e-8):
                            valid = False
                            break
                        V[j*p+k, :] = v
                    except Exception:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Snap to half-integers
            U_r = snap_half(U)
            V_r = snap_half(V)
            W_r = snap_half(W)
            
            if verify(U_r, V_r, W_r):
                return (U_r, V_r, W_r)
            
            if verify(U, V, W):
                return (U, V, W)
        
        return None

    def try_decomposition_strassen_like(r, rng):
        """Try Strassen-like block decomposition."""
        nm = n * m
        mp = m * p
        pn = p * n
        
        for attempt in range(100):
            if time.time() > deadline - 5:
                return None
            
            # Initialize with Strassen-like patterns
            U = np.zeros((nm, r), dtype=float)
            V = np.zeros((mp, r), dtype=float)
            W = np.zeros((pn, r), dtype=float)
            
            # Assign random structured values
            for c in range(r):
                # Random sign pattern
                signs_u = rng.choice([-1, 1], size=nm)
                signs_w = rng.choice([-1, 1], size=pn)
                U[:, c] = signs_u * rng.choice([0.5, 1.0], size=nm)
                W[:, c] = signs_w * rng.choice([0.5, 1.0], size=pn)
            
            # Solve for V
            valid = True
            for j in range(m):
                for k in range(p):
                    B = np.array([
                        [U[i*m+j, c] * W[k*n+i, c] for c in range(r)]
                        for i in range(n)
                    ], dtype=float)
                    rhs = np.ones(n)
                    try:
                        v = np.linalg.lstsq(B, rhs, rcond=None)[0]
                        if not np.allclose(B @ v, rhs, atol=1e-8):
                            valid = False
                            break
                        V[j*p+k, :] = v
                    except Exception:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            U_r = snap_half(U)
            V_r = snap_half(V)
            W_r = snap_half(W)
            
            if verify(U_r, V_r, W_r):
                return (U_r, V_r, W_r)
            
            if verify(U, V, W):
                return (U, V, W)
        
        return None

    best_result = None
    best_rank = rank_schoolbook
    
    # Try ranks from 45 down to 30
    for r in range(45, 29, -1):
        if time.time() > deadline - 10:
            break
        
        rng = np.random.RandomState(r * 1234 + 5678)
        
        # Try multiple strategies
        result = try_decomposition(r, rng)
        if result is None:
            result = try_decomposition_strassen_like(r, rng)
        
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best_result = (U, V, W)
                best_rank = r
                break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Fallback: schoolbook decomposition
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
