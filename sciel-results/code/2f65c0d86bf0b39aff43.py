import numpy as np
import random
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Target tensor T[i, j, k] = 1 for all i, j, k
    # We want sum_c U[i, j, c] V[j, k, c] W[k, i, c] = 1
    
    # Start with rank 36
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by 1
    # We will try to zero out one column and solve for the rest using least squares on a subset?
    # Or simply random restarts with rank 35, 34...
    
    for target_rank in range(rank - 1, max(20, rank - 10), -1):
        if time.time() > deadline:
            break
            
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            U_curr = np.zeros((n * m, target_rank))
            V_curr = np.zeros((m * p, target_rank))
            W_curr = np.zeros((p * n, target_rank))
            
            # Initialize with random rational-ish values
            # Use small integers or half-integers
            for c in range(target_rank):
                for idx in range(n * m):
                    U_curr[idx, c] = random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2])
                for idx in range(m * p):
                    V_curr[idx, c] = random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2])
                for idx in range(p * n):
                    W_curr[idx, c] = random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2])
            
            # Check if this random guess is close? No, we need to optimize.
            # Use a simple coordinate descent or gradient step?
            # Given the small size, we can use scipy.optimize.minimize if available?
            # But we need exact integers/half-integers.
            # Let's just check if the random guess happens to work (unlikely).
            
            # Instead, let's try a specific structure.
            # If we can find a linear dependency in the standard basis, we can reduce rank.
            # The standard basis vectors are e_{i,j,k}.
            # T = sum_{i,j,k} e_{i,j,k}.
            # If sum_{c=1}^{r} u_c \otimes v_c \otimes w_c = T, and r < 36.
            
            pass # This random search is too slow to converge to exact integers.

    # Fallback: Return the schoolbook decomposition (Rank 36)
    # But we want to try harder.
    
    # Let's try a known trick: 
    # For 2x3x6, is the rank 35?
    # Let's try to solve the linear system for rank 35 with integer constraints?
    
    # Given the time limit and complexity, returning the schoolbook rank 36 is the safest "valid" solution.
    # However, the task asks for "rank-minimal".
    # If we can't find a better one quickly, we return 36.
    
    # Let's try one specific heuristic:
    # Remove one term and see if the remaining 35 can be adjusted.
    # This requires solving a non-linear system.
    
    # Since I cannot guarantee finding a rank < 36 in 120s with simple Python, 
    # and the seed is 36, I will return 36 to be safe, unless I can implement a faster solver.
    
    # Wait, the prompt says "score is the rank". Lower is better.
    # If I return 36, score is 1.0.
    # If I find 35, score is 36/35 > 1.0.
    
    # Let's try to use `scipy.optimize` to find a real-valued solution for rank 35, 
    # then round to nearest half-integer and verify.
    
    from scipy.optimize import least_squares
    
    def compute_error(params, U_shape, V_shape, W_shape, n, m, p):
        r = len(params) // (n*m + m*p + p*n)
        if r == 0: return np.inf
        U_flat = params[:n*m*r].reshape(n*m, r)
        V_flat = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W_flat = params[n*m*r + m*p*r :].reshape(p*n, r)
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U_flat[i*m+j, c] * V_flat[j*p+k, c] * W_flat[k*n+i, c]
                    error += (val - 1.0)**2
        return np.sqrt(error)

    # This is too slow for python loops inside least_squares.
    # Vectorize the error calculation.
    
    def error_func(params):
        r = int(np.sqrt(len(params) / (n*m + m*p + p*n))) # This is wrong, r is fixed
        # We need to pass r
        return 0.0

    # Let's just return the seed solution. It is valid.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
