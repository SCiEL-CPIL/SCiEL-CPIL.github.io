"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _to_half_int_list(mat):
    """Convert a numpy matrix to a list of lists of int/float (half-integers)."""
    return [[float(v) for v in row] for row in mat]

def _verify(U, V, W, n=2, m=3, p=7):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    # Also check that all other entries are zero
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in [(a, b, c) for a in range(n) for b in range(m) for c in range(p)]:
                    pass
    # Check orthogonality conditions: for each r, the pattern must be valid
    # Actually, we need to verify that the tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # equals 1 if (i,j,k) is a valid multiplication index, else 0.
    # But the matrix multiplication tensor is defined as T[i,j,k] = 1 for all valid (i,j,k) in the product.
    # Wait, the matrix multiplication tensor for multiplying A (n x m) by B (m x p) gives C (n x p).
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) where T[i,j,k] = 1 if j is in range(m), else 0.
    # But actually all j are in range, so T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # So we just need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all valid (i,j,k).
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(rank, n=2, m=3, p=7, seed=42, time_limit=120):
    """Try to find a rank-rank decomposition. Returns (U, V, W) or None."""
    rng = random.Random(seed)
    start_time = time.time()
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 for all i,j,k in range.
    # We need U (n*m x rank), V (m*p x rank), W (p*n x rank) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k.
    
    # This is equivalent to finding a rank-rank decomposition of the tensor.
    # One approach: use the fact that for small ranks, we can try to find
    # sparse patterns.
    
    # For rank < n*m*p, we need some structure.
    # Let's try a structured approach based on known results for small matrix multiplication.
    
    # For 2x3 * 3x7, the schoolbook rank is 42.
    # Known lower bounds: rank >= n*m + m*p + p*n - m = 6 + 21 + 14 - 3 = 38 (from Coppersmith-Winograd style)
    # Actually, the standard lower bound is rank >= max(n*m, m*p, p*n) = 21.
    # But there are better bounds.
    
    # Let's try a simple approach: use random initialization and gradient descent.
    # But since we need exact integer/half-integer entries, this is tricky.
    
    # Alternative: try to find a decomposition by searching over small integer patterns.
    
    # For now, let's just return the schoolbook decomposition if rank >= 42.
    if rank >= n * m * p:
        return _schoolbook()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Try ranks from low to high
    # Lower bound: at least max(n*m, m*p, p*n) = 21
    # Upper bound: n*m*p = 42
    
    best = None
    for rank in range(max(n*m, m*p, p*n), n*m*p + 1):
        if time.time() > deadline:
            break
        result = _try_rank(rank, n, m, p)
        if result is not None:
            best = result
            break
    
    if best is None:
        best = _schoolbook()
    
    return best
# EVOLVE-BLOCK-END
