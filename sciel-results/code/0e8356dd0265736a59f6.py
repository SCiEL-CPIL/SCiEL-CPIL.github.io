"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # A: n x m, B: m x p, C: n x p
    dim_A = n * m  # 9
    dim_B = m * p  # 15
    dim_C = n * p  # 15
    
    # Build target tensor T[a, b, c] = 1 if C[c] depends on A[a] and B[b]
    # a = i * m + j, b = j * p + k, c = k * n + i
    T = np.zeros((dim_A, dim_B, dim_C), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Schoolbook solution: rank = n*m*p = 45
    rank_sb = n * m * p
    U_sb = np.zeros((dim_A, rank_sb))
    V_sb = np.zeros((dim_B, rank_sb))
    W_sb = np.zeros((dim_C, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[c, col] = 1
                col += 1
    
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_sb
    
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Precompute flattened tensor for each c
    T_flat = T.reshape(dim_A * dim_B, dim_C)
    
    # Search for lower rank decompositions
    # We'll try R from rank_sb-1 down to some minimum
    # For each R, we'll use a structured approach:
    # 1. Initialize U and V with sparse random patterns
    # 2. Solve for W using least squares
    # 3. Check if the decomposition is exact
    
    # Known lower bound for matrix multiplication rank:
    # For (3,3,5), the rank is at least 24 (from theoretical bounds)
    # But we'll search from 44 down
    
    for R in range(rank_sb - 1, 15, -1):
        if time.time() > deadline:
            break
        
        # Number of restarts depends on time available
        time_remaining = deadline - time.time()
        restarts = max(1, int(time_remaining / 2))
        restarts = min(restarts, 50)
        
        found = False
        
        for _ in range(restarts):
            if time.time() > deadline:
                break
            
            # Random initialization with structure
            # Use sparse patterns to encourage low-rank solutions
            U = np.zeros((dim_A, R))
            V = np.zeros((dim_B, R))
            
            # For each rank component, assign a subset of rows
            for r in range(R):
                # Randomly select which rows are non-zero for U and V
                # Use a bias towards structured patterns
                if random.random() < 0.7:
                    # Structured: select a block of rows
                    u_start = random.randint(0, dim_A - 1)
                    v_start = random.randint(0, dim_B - 1)
                    u_len = random.randint(1, min(3, dim_A - u_start))
                    v_len = random.randint(1, min(5, dim_B - v_start))
                    
                    for u_idx in range(u_start, min(u_start + u_len, dim_A)):
                        U[u_idx, r] = random.choice([-2, -1, 1, 2])
                    for v_idx in range(v_start, min(v_start + v_len, dim_B)):
                        V[v_idx, r] = random.choice([-2, -1, 1, 2])
                else:
                    # Random sparse
                    for u_idx in range(dim_A):
                        if random.random() < 0.3:
                            U[u_idx, r] = random.choice([-2, -1, 1, 2])
                    for v_idx in range(dim_B):
                        if random.random() < 0.3:
                            V[v_idx, r] = random.choice([-2, -1, 1, 2])
            
            # Solve for W
            W = np.zeros((dim_C, R))
            valid = True
            
            for c_idx in range(dim_C):
                # Build the product matrix: Prod[ab, r] = U[a, r] * V[b, r]
                # Using broadcasting
                Prod = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(dim_A * dim_B, R)
                
                # Solve Prod @ W[c_idx, :] = T_flat[:, c_idx]
                try:
                    W_row, residuals, rank, sv = np.linalg.lstsq(Prod, T_flat[:, c_idx], rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Round to nearest 0.5
                W_row_rounded = np.round(W_row * 2) / 2.0
                W[c_idx, :] = W_row_rounded
                
                # Check exact match
                if not np.allclose(Prod @ W_row_rounded, T_flat[:, c_idx], atol=1e-9):
                    valid = False
                    break
            
            if valid:
                # Verify entire tensor
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                if np.array_equal(T_recon, T):
                    best_U = U
                    best_V = V
                    best_W = W
                    best_rank = R
                    found = True
                    break
        
        if found:
            continue
        else:
            # If we can't find a solution at this rank, try a few more R values
            # but don't spend too much time
            if time.time() > deadline - 20:
                break
    
    # Convert to lists of lists with integer/half-integer values
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
# EVOLVE-BLOCK-END
