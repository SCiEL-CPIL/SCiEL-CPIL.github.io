import numpy as np
import time
import random

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
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
                
    start_time = time.time()
    deadline = start_time + 110.0

    # Precompute indices for verification
    # We need to check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # Let's create arrays of indices
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    target_vals = np.ones(len(u_idx))

    def verify_fast(U, V, W, R):
        # U, V, W are numpy arrays of shape (dim, R)
        # Compute C = sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
        # We can do this by iterating r, or using einsum if we reshape.
        # Iterating R times is O(R * N) where N = n*m*p = 96.
        # This is very fast.
        C = np.zeros(len(u_idx))
        for r in range(R):
            C += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
        return np.allclose(C, target_vals, atol=1e-9)

    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    # Search for lower rank
    # We try ranks from best_rank - 1 down to a lower bound
    # Lower bound is max(n*m, m*p, p*n) = 32
    # But let's go down to 50 to be safe and fast
    lower_bound = max(n*m, m*p, p*n)
    
    # Use a set of half-integer values for random sampling
    # {0, 0.5, 1, -0.5, -1}
    vals = [0, 0.5, 1.0, -0.5, -1.0]
    
    for R in range(best_rank - 1, lower_bound - 1, -1):
        if time.time() > deadline:
            break
            
        # Try a number of random candidates
        # The number of trials depends on remaining time
        trials = 0
        while time.time() < deadline - 5.0:
            trials += 1
            # Generate random candidate
            U_rand = np.array([[random.choice(vals) for _ in range(R)] for _ in range(n*m)])
            V_rand = np.array([[random.choice(vals) for _ in range(R)] for _ in range(m*p)])
            W_rand = np.array([[random.choice(vals) for _ in range(R)] for _ in range(p*n)])
            
            if verify_fast(U_rand, V_rand, W_rand, R):
                best_rank = R
                best_U = U_rand
                best_V = V_rand
                best_W = W_rand
                break # Found one for this R, try lower R
                
    # Convert best to list of lists with clean half-integers
    def clean_matrix(mat):
        clean = []
        for row in mat:
            clean_row = []
            for val in row:
                # Snap to nearest half-integer
                snapped = round(val * 2) / 2.0
                if abs(snapped - round(snapped)) < 1e-9:
                    clean_row.append(int(round(snapped)))
                else:
                    clean_row.append(snapped)
            clean.append(clean_row)
        return clean

    return {
        "U": clean_matrix(best_U),
        "V": clean_matrix(best_V),
        "W": clean_matrix(best_W)
    }
