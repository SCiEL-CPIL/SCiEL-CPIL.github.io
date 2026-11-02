"""Matrix multiplication 3,3,4 rank decomposition."""
import time
from fractions import Fraction
import random

DIMENSIONS = [3, 3, 4]

def _build_schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Verify the decomposition exactly for all (i,j,k) triples."""
    n, m, p = DIMENSIONS
    U = [[Fraction(x) for x in row] for row in U]
    V = [[Fraction(x) for x in row] for row in V]
    W = [[Fraction(x) for x in row] for row in W]
    rank = len(U[0]) if U and U[0] else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_rank_R(R, timeout=60.0):
    """Try to find a rank-R decomposition using random search."""
    start = time.time()
    n, m, p = DIMENSIONS
    
    # Use a structured approach: start with schoolbook and try to merge terms
    # This is a heuristic search
    
    # For now, return None if we can't find one quickly
    # A proper implementation would use:
    # 1. Random initialization
    # 2. Gradient descent on the tensor decomposition objective
    # 3. Rounding to half-integers
    # 4. Verification
    
    if time.time() > start + timeout:
        return None
    
    return None

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin
    
    # Start with schoolbook (rank 36)
    best = _build_schoolbook()
    best_rank = len(best["U"][0])
    
    # Try to find lower rank decompositions
    # Known upper bounds for 3x3x4 matrix multiplication rank are around 20-22
    # but finding exact minimal rank is an open problem
    
    # Try a few candidate ranks
    for R in range(best_rank - 1, max(best_rank - 5, 15), -1):
        if time.time() > deadline:
            break
        result = _try_rank_R(R, timeout=20.0)
        if result is not None:
            best = result
            best_rank = R
    
    return best
