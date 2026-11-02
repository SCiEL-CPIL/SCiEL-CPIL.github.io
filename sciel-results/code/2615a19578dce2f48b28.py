import numpy as np
import time
import random
import sys

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Precompute Target Tensor Indices
    # We store (u, v, w) indices that should be 1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_indices.append((u, v, w))
                
    # We also need to check non-target indices to ensure they are 0
    # The total number of indices is size_u * size_v * size_w
    # We can check all of them for exactness, but for scoring during search, 
    # we might sample or use a fast check.
    # Given small size (2304), we can check all.
    
    all_indices = [(u, v, w) for u in range(size_u) for v in range(size_v) for w in range(size_w)]
    target_set = set(target_indices)
    
    # Schoolbook Solution (Rank 48)
    R_school = n * m * p
    U_school = np.zeros((size_u, R_school), dtype=int)
    V_school = np.zeros((size_v, R_school), dtype=int)
    W_school = np.zeros((size_w, R_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 2 # Scaled by 2
                V_school[j * p + k, col] = 2
                W_school[k * n + i, col] = 2
                col += 1
                
    # Function to compute error score for integer matrices (scaled by 2)
    # Target sum is 8 for target indices, 0 otherwise.
    def compute_error(U, V, W):
        # U, V, W are integer arrays
        # Contraction: T[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
        # We only need to check if T[u, v, w] == 8 for target, 0 for others.
        # To speed up, we can compute the full tensor once?
        # np.einsum is fast.
        T = np.einsum('ur,vr,wr->uvw', U, V, W, optimize=True)
        
        error = 0
        # Check target indices
        for u, v, w in target_indices:
            if T[u, v, w] != 8:
                error += (T[u, v, w] - 8) ** 2
        # Check non-target indices (sample or all)
        # For small size, check all
        for u, v, w in all_indices:
            if (u, v, w) not in target_set:
                if T[u, v, w] != 0:
                    error += T[u, v, w] ** 2
        return error

    # Local Search Step
    def local_search(U, V, W, max_steps=10000, time_limit=5.0):
        current_error = compute_error(U, V, W)
        if current_error == 0:
            return U, V, W, 0
        
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        start_ls = time.time()
        steps = 0
        while steps < max_steps and (time.time() - start_ls) < time_limit and current_error > 0:
            steps += 1
            if time.time() > deadline - 2:
                break
                
            # Pick a random variable
            mat_choice = random.randint(0, 2)
            if mat_choice == 0:
                idx = random.randint(0, size_u - 1)
                r_idx = random.randint(0, U.shape[1] - 1)
                old_val = U[idx, r_idx]
                new_val = old_val + random.choice([-2, -1, 1, 2])
                U[idx, r_idx] = new_val
                new_error = compute_error(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                else:
                    U[idx, r_idx] = old_val
            elif mat_choice == 1:
                idx = random.randint(0, size_v - 1)
                r_idx = random.randint(0, V.shape[1] - 1)
                old_val = V[idx, r_idx]
                new_val = old_val + random.choice([-2, -1, 1, 2])
                V[idx, r_idx] = new_val
                new_error = compute_error(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                else:
                    V[idx, r_idx] = old_val
            else:
                idx = random.randint(0, size_w - 1)
                r_idx = random.randint(0, W.shape[1] - 1)
                old_val = W[idx, r_idx]
                new_val = old_val + random.choice([-2, -1, 1, 2])
                W[idx, r_idx] = new_val
                new_error = compute_error(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                else:
                    W[idx, r_idx] = old_val
                    
        return U, V, W, current_error

    # Convert integer matrices to list of half-integers
    def to_half_int_list(U, V, W):
        return {
            "U": (U / 2.0).tolist(),
            "V": (V / 2.0).tolist(),
            "W": (W / 2.0).tolist()
        }

    best_solution = to_half_int_list(U_school, V_school, W_school)
    best_rank = R_school
    
    # Search for lower ranks
    # We try ranks from R_school - 1 down to some limit
    # We use a time budget for each rank attempt
    time_per_rank = 10.0
    
    for R in range(R_school - 1, 15, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few restarts for this rank
        restarts = 5
        for restart in range(restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialization
            # Strategy 1: Random subset of schoolbook columns + noise
            # Strategy 2: Random Gaussian integers
            if restart % 2 == 0:
                indices = np.random.choice(R_school, R, replace=False)
                U_init = U_school[:, indices].copy()
                V_init = V_school[:, indices].copy()
                W_init = W_school[:, indices].copy()
                # Add small integer noise
                noise = np.random.randint(-1, 2, size=(size_u, R))
                U_init += noise
                noise = np.random.randint(-1, 2, size=(size_v, R))
                V_init += noise
                noise = np.random.randint(-1, 2, size=(size_w, R))
                W_init += noise
            else:
                U_init = np.random.randint(-2, 3, size=(size_u, R))
                V_init = np.random.randint(-2, 3, size=(size_v, R))
                W_init = np.random.randint(-2, 3, size=(size_w, R))
                
            # Run local search
            U_opt, V_opt, W_opt, final_error = local_search(U_init, V_init, W_opt if 'W_opt' in locals() else W_init, max_steps=5000, time_limit=time_per_rank)
            
            if final_error == 0:
                # Verify exactness one more time
                if compute_error(U_opt, V_opt, W_opt) == 0:
                    best_rank = R
                    best_solution = to_half_int_list(U_opt, V_opt, W_opt)
                    break # Move to next lower rank
                    
    return best_solution

# EVOLVE-BLOCK-END
