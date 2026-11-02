"""Tensor rank minimization for 2x3 by 3x4 matrix multiplication via random search + structured repair."""
import random
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import sys

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_random_decomposition(deadline):
    """Try to find a decomposition of a given rank via random structured search."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # The multiplication tensor T[i,j,k] = 1 for all (i,j,k) in [0,n)x[0,m)x[0,p)
    # We want T = sum_c U[:,c] (x) V[:,c] (x) W[:,c]
    # i.e., for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Known lower bounds suggest rank is around 14-16 for this size.
    # Let's try ranks from low to high, with random + local search.
    
    best_result = _schoolbook()
    best_rank = best_result['U'] and len(best_result['U'][0]) or nm*mp*pn
    
    for target_rank in [12, 13, 14, 15, 16, 17, 18]:
        if time.time() > deadline - 20:
            break
        # Try a few random initializations
        attempts = 0
        max_attempts = 200
        while attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            # Random matrices with entries from a small set
            U = [[random.choice([-1, 0, 1, Fraction(1,2), -Fraction(1,2)]) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.choice([-1, 0, 1, Fraction(1,2), -Fraction(1,2)]) for _ in range(target_rank)] for _ in range(mp)]
            W = [[random.choice([-1, 0, 1, Fraction(1,2), -Fraction(1,2)]) for _ in range(target_rank)] for _ in range(pn)]
            
            # Verify: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1
            # This is expensive, so let's use a faster check with numpy if available, else fall back
            try:
                import numpy as np
                U_arr = np.array(U, dtype=object)
                V_arr = np.array(V, dtype=object)
                W_arr = np.array(W, dtype=object)
                # Build the tensor via einsum
                # Actually let's just do it directly since dimensions are small
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = Fraction(0)
                            for c in range(target_rank):
                                s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                            if s != Fraction(1):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_result = {"U": U, "V": V, "W": W}
                    # Found a valid one at this rank, try to go lower
                    break
            except Exception:
                # Fallback without numpy
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = Fraction(0)
                            for c in range(target_rank):
                                s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                            if s != Fraction(1):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_result = {"U": U, "V": V, "W": W}
                    break
        if target_rank < best_rank:
            # Found improvement, continue trying lower
            continue
        else:
            # If we tried multiple attempts without success, try a higher rank
            if time.time() > deadline - 30:
                break
    
    return best_result

def construct():
    deadline = time.time() + 130  # Safety margin: evaluator kills at 150s
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = len(best['U'][0])
    
    # Try to find better via structured search
    try:
        result = _try_random_decomposition(deadline)
        r = len(result['U'][0])
        if r < best_rank:
            best = result
            best_rank = r
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
