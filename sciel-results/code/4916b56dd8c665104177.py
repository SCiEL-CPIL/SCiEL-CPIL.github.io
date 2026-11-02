import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
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
    deadline = start_time + 110 # 110s limit to be safe
    
    # Pre-compute target tensor indices and values
    # T[alpha, beta, gamma] = 1 if j_alpha == j_beta, else 0
    # alpha = i*m + j
    # beta = j'*p + k
    # gamma = k*n + i
    # Note: gamma depends on k and i.
    # The condition for non-zero is j == j'.
    # Also, for a specific (i, j, k), gamma is fixed.
    # So T[alpha, beta, gamma] is 1 if j_alpha == j_beta AND gamma matches (k,i) derived from alpha, beta?
    # Actually, gamma is an independent index in the tensor definition.
    # The tensor is T_{\alpha \beta \gamma}.
    # T_{\alpha \beta \gamma} = 1 if j_alpha == j_beta AND i_alpha == i_gamma AND k_beta == k_gamma.
    # Let's verify.
    # alpha = i*m + j.
    # beta = j'*p + k.
    # gamma = k'*n + i'.
    # We need sum_r U_{alpha,r} V_{beta,r} W_{gamma,r} = 1 if (i,j) and (j',k) and (k',i') form a valid chain.
    # Valid chain: j == j' AND k == k' AND i == i'.
    # So T is 1 only for the specific triples where indices match perfectly.
    # All other triples are 0.
    
    # We will check all 2500 entries.
    # Create arrays of indices for valid and invalid entries?
    # Or just compute the whole tensor and compare.
    
    # Target tensor is sparse.
    # We can create a target array of zeros and set the valid ones to 1.
    T_target = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_target[alpha, beta, gamma] = 1

    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try ranks from 49 down to 30
    # Start from 49 to find improvements quickly
    for R in range(49, 29, -1):
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        max_attempts = 200 # More attempts due to fast check
        
        while not found and attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Random integers -2, -1, 0, 1, 2
            # Scale might be needed, but small integers are good for exactness
            U = np.random.randint(-2, 3, (n * m, R))
            V = np.random.randint(-2, 3, (m * p, R))
            W = np.random.randint(-2, 3, (p * n, R))
            
            # Compute T_calc
            # T_calc[alpha, beta, gamma] = sum_r U[alpha, r] * V[beta, r] * W[gamma, r]
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            
            # Check if T_calc matches T_target
            if np.array_equal(T_calc, T_target):
                found = True
                best_rank = R
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break # Found a solution for this R, try lower R?
                # The loop goes 49, 48...
                # If we find 49, we break inner loop.
                # Outer loop continues to 48.
                # This is correct.
        
        # If we found a solution for R, we continue to R-1.
        # If we don't, we continue to R-1.
        
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
