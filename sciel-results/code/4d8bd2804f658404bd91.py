import time
import numpy as np
import random

DIMENSIONS = [3, 4, 6]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _is_half_integer(val):
    """Check if val is a half-integer (multiple of 0.5)."""
    v = round(val * 2)
    return abs(val - v/2.0) < 1e-9

def _to_half_integer(val):
    """Round to nearest half-integer."""
    return round(val * 2) / 2.0

def _try_rank(r, n, m, p, deadline):
    """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
    num_U = n * m
    num_V = m * p
    num_W = p * n
    
    # The tensor has n*m*p = 72 entries, all equal to 1.
    # Variables: r*(num_U + num_V + num_W) = r*(12+24+18) = 54r
    # For r=72, we have 3888 variables for 72 equations (overdetermined but solvable)
    # For lower r, we need to check feasibility.
    
    # Try random initialization with small half-integer values
    candidates = [0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2]
    
    for attempt in range(200):
        if time.time() > deadline:
            return None
        
        U = [[random.choice(candidates) for _ in range(r)] for _ in range(num_U)]
        V = [[random.choice(candidates) for _ in range(r)] for _ in range(num_V)]
        W = [[random.choice(candidates) for _ in range(r)] for _ in range(num_W)]
        
        if _verify(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # We know rank must be at least max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But realistically, for matrix multiplication, the rank is often close to n*m*p
    # for rectangular cases without special structure.
    
    # Try ranks from 71 down to some lower bound
    # Given time constraints, try a few promising ranks
    for r in range(best_rank - 1, 23, -1):
        if time.time() > deadline - 5:
            break
        result = _try_rank(r, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            # Verify once more
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return best

# EVOLVE-BLOCK-END
