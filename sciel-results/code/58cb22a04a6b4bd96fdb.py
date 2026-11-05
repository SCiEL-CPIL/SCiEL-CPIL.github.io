import numpy as np
import time
from fractions import Fraction
import random
from itertools import combinations

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school

    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Target ranks to try
    # Known lower bound is 22, schoolbook is 36
    for target_rank in range(22, 36):
        if time.time() > deadline:
            break
            
        # Try to find a decomposition of this rank
        found = False
        max_restarts = 5000
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize U and V with random small integers
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            success = True
            # Solve for W for each (k, i) pair
            for k in range(p):
                for i in range(n):
                    # Construct M (m x target_rank)
                    # M[j, r] = U[i*m + j, r] * V[j*p + k, r]
                    M = np.zeros((m, target_rank), dtype=int)
                    for j in range(m):
                        for r in range(target_rank):
                            M[j, r] = U[i * m + j][r] * V[j * p + k][r]
                    
                    # We want M y = 2 * 1 (vector of 2s) with y in Z^target_rank
                    # Try subsets of columns of size m
                    found_solution = False
                    # Optimization: If M is all zeros, fail
                    if not np.any(M):
                        success = False
                        break
                        
                    # Iterate through combinations of m columns
                    # To speed up, we can shuffle columns first
                    cols = list(range(target_rank))
                    random.shuffle(cols)
                    
                    # Limit the number of subsets to check to avoid timeout
                    # If target_rank is large, binom(R, m) is large.
                    # For R=20, m=3, it's ~1140.
                    # For R=30, m=3, it's ~4060.
                    # We can check up to 1000 subsets.
                    max_subsets_to_check = 2000
                    checked = 0
                    
                    for comb in combinations(cols, m):
                        if checked >= max_subsets_to_check:
                            break
                        checked += 1
                        M_sub = M[:, list(comb)]
                        # Solve M_sub y_sub = [2, 2, ..., 2]
                        try:
                            # Use numpy to solve
                            b = np.ones(m) * 2
                            y_sub = np.linalg.solve(M_sub, b)
                            # Check if y_sub is integer
                            if np.allclose(y_sub, np.round(y_sub), atol=1e-9):
                                y = np.zeros(target_rank, dtype=int)
                                for idx, col_idx in enumerate(comb):
                                    y[col_idx] = int(np.round(y_sub[idx]))
                                # Verify
                                if np.allclose(M @ y, b, atol=1e-9):
                                    # Found valid y
                                    for r in range(target_rank):
                                        W[k * n + i][r] = Fraction(y[r], 2)
                                    found_solution = True
                                    break
                        except np.linalg.LinAlgError:
                            continue
                    
                    if not found_solution:
                        success = False
                        break

            if success:
                # Valid decomposition found
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                found = True
                break
        
        if found:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
