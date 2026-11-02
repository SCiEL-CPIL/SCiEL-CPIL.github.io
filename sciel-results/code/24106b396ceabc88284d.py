"""
Matrix multiplication 3x3 * 3x4: rank-minimal exact tensor decomposition.

The seed program uses the schoolbook decomposition of rank n*m*p = 36.
We search for a lower-rank decomposition using integer/half-integer entries.

The task is to find matrices U (9 x r), V (12 x r), W (12 x r) such that
for all (i, j, k) in [3]x[3]x[4]:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

where n=3, m=3, p=4.
"""

import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS
RANK_TARGET = 20  # Known upper bound for matmul(3,3,4) is around 20-22
MAX_RANK_TRY = 25
DEADLINE = 145.0  # safety margin before 150s kill
START_TIME = time.time()

def remaining_time():
    return DEADLINE - (time.time() - START_TIME)

def check_decomposition(U, V, W):
    """Check if U, V, W exactly decompose the all-ones tensor."""
    r = U.shape[1]
    # Build the contraction
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # U is (n*m, r), V is (m*p, r), W is (p*n, r)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-6:
                    return False
    return True

def try_rank(r, time_budget=30.0):
    """Try to find a decomposition of rank r using random search with local refinement."""
    if remaining_time() < time_budget:
        return None
    
    U_rows = n * m  # 9
    V_rows = m * p  # 12
    W_rows = p * n  # 12
    
    for attempt in range(200):
        if remaining_time() < 5:
            break
        
        # Random initialization with integers in {-2, -1, 0, 1, 2}
        U = np.random.randint(-2, 3, size=(U_rows, r))
        V = np.random.randint(-2, 3, size=(V_rows, r))
        W = np.random.randint(-2, 3, size=(W_rows, r))
        
        # Local refinement: coordinate descent
        for it in range(500):
            if remaining_time() < 5:
                break
            # Pick a random element to update
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                idx = random.randint(0, U_rows - 1)
                c = random.randint(0, r - 1)
                # Try all integer values in range
                best_val = U[idx, c]
                best_err = float('inf')
                for val in range(-3, 4):
                    U[idx, c] = val
                    # Compute error (simplified: just check a few points)
                    err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for cc in range(r):
                                    s += U[i*m+j, cc] * V[j*p+k, cc] * W[k*n+i, cc]
                                err += (s - 1) ** 2
                    if err < best_err:
                        best_err = err
                        best_val = val
                    if err == 0:
                        break
                U[idx, c] = best_val
            elif mat == 'V':
                idx = random.randint(0, V_rows - 1)
                c = random.randint(0, r - 1)
                best_val = V[idx, c]
                best_err = float('inf')
                for val in range(-3, 4):
                    V[idx, c] = val
                    err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for cc in range(r):
                                    s += U[i*m+j, cc] * V[j*p+k, cc] * W[k*n+i, cc]
                                err += (s - 1) ** 2
                    if err < best_err:
                        best_err = err
                        best_val = val
                    if err == 0:
                        break
                V[idx, c] = best_val
            else:
                idx = random.randint(0, W_rows - 1)
                c = random.randint(0, r - 1)
                best_val = W[idx, c]
                best_err = float('inf')
                for val in range(-3, 4):
                    W[idx, c] = val
                    err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for cc in range(r):
                                    s += U[i*m+j, cc] * V[j*p+k, cc] * W[k*n+i, cc]
                                err += (s - 1) ** 2
                    if err < best_err:
                        best_err = err
                        best_val = val
                    if err == 0:
                        break
                W[idx, c] = best_val
            
            if best_err == 0:
                # Convert to list-of-lists format
                return [U.tolist(), V.tolist(), W.tolist()]
    
    return None

def construct():
    # Start with schoolbook (rank 36) as fallback
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    # Try to find lower rank
    for r in range(16, MAX_RANK_TRY + 1):
        if remaining_time() < 10:
            break
        result = try_rank(r, time_budget=min(remaining_time() - 5, 40))
        if result is not None:
            U, V, W = result
            # Final verification
            if check_decomposition(np.array(U), np.array(V), np.array(W)):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_fb, "V": V_fb, "W": W_fb}
