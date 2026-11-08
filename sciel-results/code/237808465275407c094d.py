"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

_start_time = time.time()

def _construct_rank_55():
    """
    Hardcoded rank-55 decomposition for 2x3x10 matrix multiplication.
    Based on 5 blocks of 2 columns, each using 11 ranks.
    """
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate over 5 blocks of 2 columns
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1: (A00 + A11) * (B0,k0 + B1,k1)
        U[0 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M2: (A10 + A11) * B0,k0
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1
        
        # M3: A00 * (B0,k1 - B1,k1)
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M4: A11 * (B1,k0 - B0,k0)
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # M5: (A00 + A01) * B1,k1
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # M6: (A10 - A00) * (B0,k0 + B0,k1)
        U[1 * 3 + 0][col] += 1
        U[0 * 3 + 0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M7: (A01 - A11) * (B1,k0 + B1,k1)
        U[0 * 3 + 1][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # W00: A02 * B2,k0
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # W01: A02 * B2,k1
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # W10: A12 * B2,k0
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # W11: A12 * B2,k1
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def _drop_column(U, V, W, col, keep_cols):
    """
    Try to drop column `col` by expressing W[:,col] as a combination of
    W[:,keep_cols]. Returns (new_U, new_V, new_W) or None.
    """
    Wc = W[:, col]
    Wk = W[:, keep_cols]
    try:
        a, res, rank, sv = np.linalg.lstsq(Wk, Wc, rcond=None)
    except Exception:
        return None
    
    if not np.allclose(Wk @ a, Wc, atol=1e-7, rtol=1e-7):
        return None
        
    new_U = np.zeros((U.shape[0], len(keep_cols)))
    new_V = np.zeros((V.shape[0], len(keep_cols)))
    new_W = np.zeros((W.shape[0], len(keep_cols)))
    for li, kc in enumerate(keep_cols):
        new_U[:, li] = U[:, kc] + a[li] * U[:, col]
        new_V[:, li] = V[:, kc] + a[li] * V[:, col]
        new_W[:, li] = W[:, kc] + a[li] * W[:, col]
    return new_U, new_V, new_W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-7):
    """
    Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_decomposition_exact(U, V, W, n, m, p):
    """
    Verify decomposition using exact integer arithmetic.
    """
    Ua = np.array(U, dtype=int)
    Va = np.array(V, dtype=int)
    Wa = np.array(W, dtype=int)
    rank = Ua.shape[1]
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if s != 1:
                    return False
    return True

def _local_improve(U, V, W, time_budget):
    """Attempt to reduce rank via null-space column elimination."""
    start = time.time()
    cur_U, cur_V, cur_W = U.copy(), V.copy(), W.copy()
    cur_cols = list(range(cur_U.shape[1]))
    improved = True
    n, m, p = DIMENSIONS
    
    while improved:
        improved = False
        if time.time() - start > time_budget:
            break
        for col in list(cur_cols):
            if time.time() - start > time_budget:
                break
            keep_cols = [c for c in cur_cols if c != col]
            if not keep_cols:
                continue
            res = _drop_column(cur_U, cur_V, cur_W, col, keep_cols)
            if res is not None:
                nU, nV, nW = res
                if _verify_decomposition(nU, nV, nW, n, m, p, tol=1e-5):
                    cur_U, cur_V, cur_W = nU, nV, nW
                    cur_cols = [c for c in cur_cols if c != col]
                    improved = True
                    break
    return cur_U, cur_V, cur_W, len(cur_cols)

def _to_int_matrices(U, V, W):
    """Convert numpy float matrices to int lists."""
    U = np.rint(U).astype(int)
    V = np.rint(V).astype(int)
    W = np.rint(W).astype(int)
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = 140.0  # safety margin

    # Step 1: Hardcoded rank-55 decomposition
    U, V, W = _construct_rank_55()
    rank = U.shape[1]
    best = (rank, U.copy(), V.copy(), W.copy())

    # Step 2: Local improvement
    if time.time() < DEADLINE:
        time_budget = max(1.0, DEADLINE - time.time() - 5.0)
        try:
            nU, nV, nW, new_rank = _local_improve(U, V, W, time_budget)
            if new_rank < best[0]:
                if _verify_decomposition(nU, nV, nW, n, m, p, tol=1e-5):
                    best = (new_rank, nU, nV, nW)
        except Exception:
            pass

    r, U, V, W = best
    U, V, W = _to_int_matrices(U, V, W)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
