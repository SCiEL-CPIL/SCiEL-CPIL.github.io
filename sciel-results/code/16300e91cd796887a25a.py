"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using numpy vectorization."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    # Reshape
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    # We want to check if sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1 for all i,j,k
    # This is equivalent to checking the tensor T = sum_c u_c \otimes v_c \otimes w_c
    # where u_c is in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    # But the indices are permuted in W.
    
    # Efficient check:
    # For each i, j, k:
    # s = dot(U[i,j,:], V[j,k,:] * W[k,i,:])
    
    # We can loop over i, j, k but vectorize the dot product.
    # Or, we can construct the full tensor and check.
    # Given small dimensions, a simple loop with numpy dot is fast enough.
    
    # Pre-compute V * W element-wise? No, W depends on k, i.
    
    # Let's just use the loop, it's only 3*5*5 = 75 iterations, each doing a dot product of length rank.
    # This is very fast.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_t[i, j, :] * V_t[j, k, :] * W_t[k, i, :]
                if abs(np.sum(prod) - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s to leave margin for 120s limit
    
    # Start with schoolbook
    U, V, W, best_rank = _schoolbook(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    
    # Candidates for entries
    candidates = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    # Strategy:
    # 1. Try to find rank 74 (1 less than schoolbook)
    # 2. Try to find rank 73
    # ...
    # Since random search is unlikely to find < 75, we focus on robustness and speed.
    # However, we will try a few ranks just in case.
    
    # We will iterate from best_rank - 1 down to a lower bound (e.g., 40)
    # But we will limit the number of attempts per rank to ensure we don't timeout.
    
    # Given the difficulty, we might not find anything better.
    # We will still run the search to be safe.
    
    # We will try ranks from 74 down to 40.
    # To maximize coverage, we will spend more time on higher ranks (closer to 75)
    # as they are more likely to be reachable by small perturbations.
    
    ranks_to_try = list(range(74, 39, -1))
    
    for target_rank in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Determine attempts based on remaining time
        # Assume 1ms per attempt (conservative estimate for verification)
        # Actually, verification is ~75 * rank * ops. For rank 75, ~5600 ops. Very fast.
        # Let's allow 1000 attempts per rank if time permits.
        
        max_attempts = 1000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            # Using random integers/half-integers
            U_r = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(n*m)]
            V_r = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(m*p)]
            W_r = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify_fast(U_r, V_r, W_r, n, m, p):
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                # If we find a solution, we can break to try lower ranks
                break
    
    return best_result
# EVOLVE-BLOCK-END
