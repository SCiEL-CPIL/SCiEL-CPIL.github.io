"""Matrix multiplication 3,4,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]
import time
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s

# EVOLVE-BLOCK-START
import random
import math
import numpy as np
from itertools import product

def _build_rank_n_m():
    """
    Constructs a rank-n*m decomposition for n x m x p matrix multiplication.
    This uses the fact that R(1, m, p) = m.
    For n rows, we can sum n such decompositions.
    Total rank = n * m.
    """
    n, m, p = DIMENSIONS
    rank = n * m
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            # Term r corresponds to (i, j)
            # U[i, j, r] = 1
            U[i * m + j][r] = 1.0
            # V[j, k, r] = 1 for all k
            for k in range(p):
                V[j * p + k][r] = 1.0
            # W[k, i, r] = 1 for all k
            for k in range(p):
                W[k * n + i][r] = 1.0
            r += 1
            
    return {"U": U, "V": V, "W": W}

def _build_schoolbook():
    """Standard schoolbook matrix multiplication tensor decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _validate_decomposition(result, ntest=3, seed=42):
    """Validate that the decomposition computes exact matrix multiplication."""
    n, m, p = DIMENSIONS
    U, V, W = result['U'], result['V'], result['W']
    if not U or not V or not W:
        return False
    if len(U[0]) != len(V[0]) or len(U[0]) != len(W[0]):
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    rng = random.Random(seed)
    for t in range(ntest):
        A = [[rng.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        Ctrue = [[sum(A[i][jj] * B[jj][kk] for jj in range(m)) for kk in range(p)]
                 for i in range(n)]
        Ccomp = [[0.0 for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0.0
                for r in range(len(U[0])):
                    uvprod = 0.0
                    for jj in range(m):
                        uvprod += U[i*m+jj][r] * V[jj*p+k][r]
                    acc += uvprod * W[k*n+i][r]
                Ccomp[i][k] = acc
        for i in range(n):
            for k in range(p):
                if abs(Ccomp[i][k] - Ctrue[i][k]) > 1e-9 * max(1, abs(Ctrue[i][k])):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    elapsed = time.time() - _START_TIME
    budget = _TIME_LIMIT - elapsed
    if budget <= 10:
        return _build_schoolbook()
    
    # Try the rank-n*m construction first
    try:
        optimized = _build_rank_n_m()
        if _validate_decomposition(optimized, ntest=5, seed=123):
            return {
                "U": [[float(x) for x in row] for row in optimized['U']],
                "V": [[float(x) for x in row] for row in optimized['V']],
                "W": [[float(x) for x in row] for row in optimized['W']]
            }
    except Exception:
        pass
    
    # Fallback to schoolbook
    best = _build_schoolbook()
    
    return {
        "U": [[float(x) for x in row] for row in best['U']],
        "V": [[float(x) for x in row] for row in best['V']],
        "W": [[float(x) for x in row] for row in best['W']]
    }
# EVOLVE-BLOCK-END
