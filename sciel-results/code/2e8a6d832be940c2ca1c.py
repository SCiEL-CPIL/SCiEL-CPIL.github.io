"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
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
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1 for all i,j,k"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_random_search(deadline, best_result):
    """Try to find a lower rank decomposition via random search."""
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Try ranks from a reasonable minimum up to schoolbook
    # Lower bound: max(nm, mp, pn) = max(16, 20, 20) = 20
    # But actual lower bound is higher. Let's try from 40 up.
    for target_rank in range(40, 81):
        if time.time() > deadline - 5:
            break
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            # Random sparse matrices with small integer entries
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(mp)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(pn)]
            
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    return None

def _try_structured_reduction(deadline, best_result):
    """Try to reduce schoolbook by merging terms where possible."""
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    rank = n * m * p
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Try to find linear dependencies to reduce rank
    # This is a complex optimization; for now return schoolbook
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook as baseline
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Try structured approaches first
    result = _try_structured_reduction(deadline, best)
    if result and len(result["U"][0]) < best_rank:
        best = result
        best_rank = len(best["U"][0])
    
    # Try random search for lower rank
    result = _try_random_search(deadline, best)
    if result and len(result["U"][0]) < best_rank:
        best = result
        best_rank = len(best["U"][0])
    
    return best
# EVOLVE-BLOCK-END
