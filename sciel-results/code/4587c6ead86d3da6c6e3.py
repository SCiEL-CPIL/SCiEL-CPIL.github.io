"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
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
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Precompute index arrays for fast evaluation
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    def check_decomposition_fast(U, V, W, rank):
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        total = np.zeros(n * m * p, dtype=np.float64)
        for r in range(rank):
            total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
        
        if np.max(np.abs(total - 1.0)) > 1e-6:
            return False
        return True

    # Strategy: Block decomposition with local search
    # Split p=8 into 3+3+2
    # Use rank-23 for 3x3x3x3 (if found) or rank-27 (schoolbook)
    # Use rank-18 for 3x3x3x2 (schoolbook)
    # Target rank: 23+23+18 = 64 or 27+27+18 = 72
    
    # We will try to find a rank-64 decomposition using a local search
    # initialized with the block structure.
    
    # Define the block structure
    # Block 1: k in [0, 2]
    # Block 2: k in [3, 5]
    # Block 3: k in [6, 7]
    
    # For the local search, we will use a rank of 64.
    # We will initialize U, V, W with zeros, then set the block structure.
    # For Block 1 and 2, we will use a sparse pattern that allows for Strassen-like reductions.
    # For Block 3, we will use schoolbook.
    
    rank_target = 64
    U = [[0] * rank_target for _ in range(n * m)]
    V = [[0] * rank_target for _ in range(m * p)]
    W = [[0] * rank_target for _ in range(p * n)]
    
    # Block 3: k in [6, 7], schoolbook (18 terms)
    # Indices for Block 3
    k_block3 = [6, 7]
    # For each i, j, k in block 3, assign a term
    col_idx = 46 # Start after 23+23
    for i in range(n):
        for j in range(m):
            for k in k_block3:
                U[i * m + j][col_idx] = 1
                V[j * p + k][col_idx] = 1
                W[k * n + i][col_idx] = 1
                col_idx += 1
                
    # Block 1: k in [0, 2], rank 23
    # Block 2: k in [3, 5], rank 23
    # We will use a random initialization for these blocks and try to solve.
    # This is a nonlinear problem, so we use a local search.
    
    # Initialize Block 1 and 2 with random values from {-1, 0, 1}
    for r in range(23):
        for i in range(n):
            for j in range(m):
                if random.random() < 0.5:
                    U[i * m + j][r] = random.choice([-1, 1])
        for j in range(m):
            for k in range(3): # k in [0, 2]
                if random.random() < 0.5:
                    V[j * p + k][r] = random.choice([-1, 1])
        for k in range(3): # k in [0, 2]
            for i in range(n):
                if random.random() < 0.5:
                    W[k * n + i][r] = random.choice([-1, 1])
                    
    for r in range(23, 46):
        for i in range(n):
            for j in range(m):
                if random.random() < 0.5:
                    U[i * m + j][r] = random.choice([-1, 1])
        for j in range(m):
            for k in range(3, 6): # k in [3, 5]
                if random.random() < 0.5:
                    V[j * p + k][r] = random.choice([-1, 1])
        for k in range(3, 6): # k in [3, 5]
            for i in range(n):
                if random.random() < 0.5:
                    W[k * n + i][r] = random.choice([-1, 1])
                    
    # Local search to satisfy the equations
    # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a system of 72 equations.
    # We have 64 variables (terms).
    # We can try to adjust the coefficients.
    
    # Since the problem is hard to solve exactly with random init, we will
    # fall back to schoolbook if we can't find a solution quickly.
    # However, we can try a few steps of gradient descent on the squared error.
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    def compute_error():
        total = np.zeros(n * m * p, dtype=np.float64)
        for r in range(rank_target):
            total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
        return np.sum((total - 1.0)**2)
    
    # Try a few random restarts with local search
    for restart in range(10):
        if time.time() > deadline - 10:
            break
            
        # Reset Block 1 and 2 to random values
        for r in range(46):
            for i in range(n * m):
                U_arr[i, r] = random.choice([-1, 0, 1])
            for i in range(m * p):
                if r < 23:
                    k_local = i % p
                    if k_local < 3:
                        V_arr[i, r] = random.choice([-1, 0, 1])
                    else:
                        V_arr[i, r] = 0
                else:
                    k_local = i % p
                    if 3 <= k_local < 6:
                        V_arr[i, r] = random.choice([-1, 0, 1])
                    else:
                        V_arr[i, r] = 0
            for i in range(p * n):
                if r < 23:
                    k_local = i // n
                    if k_local < 3:
                        W_arr[i, r] = random.choice([-1, 0, 1])
                    else:
                        W_arr[i, r] = 0
                else:
                    k_local = i // n
                    if 3 <= k_local < 6:
                        W_arr[i, r] = random.choice([-1, 0, 1])
                    else:
                        W_arr[i, r] = 0
                        
        # Simple local search: perturb one coefficient at a time
        for step in range(100):
            if time.time() > deadline - 5:
                break
                
            err = compute_error()
            if err < 1e-6:
                # Check if it's an exact integer solution
                U_int = [[int(round(x)) for x in row] for row in U_arr.tolist()]
                V_int = [[int(round(x)) for x in row] for row in V_arr.tolist()]
                W_int = [[int(round(x)) for x in row] for row in W_arr.tolist()]
                if check_decomposition_fast(U_int, V_int, W_int, rank_target):
                    best_rank = rank_target
                    best_result = {"U": U_int, "V": V_int, "W": W_int}
                    return best_result
                    
            # Pick a random matrix and element to perturb
            mat_idx = random.choice([0, 1, 2])
            if mat_idx == 0:
                row = random.randint(0, n * m - 1)
                col = random.randint(0, rank_target - 1)
                # Only perturb if it's in the active block
                if col < 46:
                    U_arr[row, col] += random.choice([-1, 1])
            elif mat_idx == 1:
                row = random.randint(0, m * p - 1)
                col = random.randint(0, rank_target - 1)
                if col < 23:
                    if row % p < 3:
                        V_arr[row, col] += random.choice([-1, 1])
                elif col < 46:
                    if 3 <= row % p < 6:
                        V_arr[row, col] += random.choice([-1, 1])
            else:
                row = random.randint(0, p * n - 1)
                col = random.randint(0, rank_target - 1)
                if col < 23:
                    if row // n < 3:
                        W_arr[row, col] += random.choice([-1, 1])
                elif col < 46:
                    if 3 <= row // n < 6:
                        W_arr[row, col] += random.choice([-1, 1])
                        
    return best_result
# EVOLVE-BLOCK-END
