import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # 100s safety margin
    
    # 1. Schoolbook Decomposition (Rank 105)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # 2. Attempt to find a lower rank via random sparse search
    # We search for R < 105
    # This is a heuristic search. It may not find a solution.
    
    # Helper to check validity
    def is_valid(U, V, W):
        # Check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        # Compute tensor T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        # We can do this efficiently by iterating r or using einsum
        # T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        # Check if T is all 1s
        
        # Optimize: only check a subset if time is tight, but for small dims, full check is fast
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=1e-9)

    # Search loop
    # Try ranks from 104 down to 90
    for R in range(rank_school - 1, 89, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations for this rank
        attempts = 0
        max_attempts = 500  # Limit attempts per rank
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Random sparse initialization
            # Use small integers or half-integers
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # Fill with random values from {-1, 0, 0.5, 1, 1.5, 2}
            # Sparse: ~20% non-zero
            for r in range(R):
                # Randomly select entries
                if random.random() < 0.5:
                    idx = random.randint(0, n*m - 1)
                    U[idx, r] = random.choice([-1, 0.5, 1, 1.5, 2])
                if random.random() < 0.5:
                    idx = random.randint(0, m*p - 1)
                    V[idx, r] = random.choice([-1, 0.5, 1, 1.5, 2])
                if random.random() < 0.5:
                    idx = random.randint(0, p*n - 1)
                    W[idx, r] = random.choice([-1, 0.5, 1, 1.5, 2])
            
            if is_valid(U, V, W):
                best_rank = R
                best_U = U
                best_V = V
                best_W = W
                break # Found a valid solution for this R, try lower R
        
    # Convert to lists of lists as required by interface
    # The interface expects integer/half-integer entries
    U_list = [[0] * best_rank for _ in range(n * m)]
    V_list = [[0] * best_rank for _ in range(m * p)]
    W_list = [[0] * best_rank for _ in range(p * n)]
    
    for i in range(n * m):
        for r in range(best_rank):
            val = best_U[i, r]
            # Ensure it's a "nice" number if possible, but numpy float is fine if it's exact
            # The prompt says "integer/half-integer". 
            # If we found a solution with random floats, it might not be exactly half-integer.
            # However, the search above only used half-integers.
            U_list[i][r] = val
            
    for j in range(m * p):
        for r in range(best_rank):
            V_list[j][r] = best_V[j, r]
            
    for k in range(p * n):
        for r in range(best_rank):
            W_list[k][r] = best_W[k, r]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
