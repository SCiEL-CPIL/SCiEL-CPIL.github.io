"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
import time
import random

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=3, m=4, p=4
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Try to find lower rank decompositions
    # The tensor has n*m*p = 48 entries, each must equal 1
    # U has shape (n*m, rank) = (12, rank)
    # V has shape (m*p, rank) = (16, rank)  
    # W has shape (p*n, rank) = (12, rank)
    
    # For each rank r, we need to solve:
    # For all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # with U,V,W having integer or half-integer entries
    
    # Try ranks from 40 down to 30 (reasonable range to explore)
    for rank in range(40, 29, -1):
        if time.time() > deadline:
            break
        
        # Try randomized search with integer/half-integer values
        # Values in {-1, -0.5, 0, 0.5, 1, 1.5, 2}
        possible_values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Randomly initialize U, V, W with small integer/half-integer values
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Set random non-zero entries
            for c in range(rank):
                # Each column c defines a rank-1 contribution
                # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
                # For the product to contribute to specific (i,j,k), we need
                # U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] to be non-zero for some (i,j,k)
                
                # Choose which indices are active for this rank-1 term
                # Simple approach: set a few entries to ±1 or ±0.5
                for _ in range(3):
                    idx = random.randint(0, n * m - 1)
                    U[idx, c] = random.choice(possible_values)
                for _ in range(3):
                    idx = random.randint(0, m * p - 1)
                    V[idx, c] = random.choice(possible_values)
                for _ in range(3):
                    idx = random.randint(0, p * n - 1)
                    W[idx, c] = random.choice(possible_values)
            
            # Check if this decomposition works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                        val = 0.0
                        for c in range(rank):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Found a valid decomposition!
                U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n * m)]
                V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m * p)]
                W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p * n)]
                
                # Verify all entries are integer or half-integer
                def is_half_int(x):
                    return abs(x * 2 - round(x * 2)) < 1e-9
                
                all_valid = True
                for row in U_list:
                    for v in row:
                        if not is_half_int(v):
                            all_valid = False
                            break
                    if not all_valid:
                        break
                if all_valid:
                    for row in V_list:
                        for v in row:
                            if not is_half_int(v):
                                all_valid = False
                                break
                        if not all_valid:
                            break
                if all_valid:
                    for row in W_list:
                        for v in row:
                            if not is_half_int(v):
                                all_valid = False
                                break
                            if not all_valid:
                                break
                
                if all_valid:
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    return best_result
    
    return best_result
# EVOLVE-BLOCK-END
