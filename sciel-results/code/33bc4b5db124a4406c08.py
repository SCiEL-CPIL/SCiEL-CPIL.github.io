import numpy as np
import random
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find a decomposition of rank 24
    # The condition is sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k
    
    def check_solution(U, V, W, rank):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Check for all i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Try rank 24
    rank = 24
    # Initialize with random small integers
    # Use a simple local search or random restarts
    start_time = time.time()
    limit = 100  # seconds
    
    best_rank = n*m*p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook as fallback
    U_school = np.zeros((n*m, n*m*p))
    V_school = np.zeros((m*p, n*m*p))
    W_school = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
                
    if check_solution(U_school, V_school, n*m*p):
        best_rank = n*m*p
        best_U = U_school
        best_V = V_school
        best_W = W_school

    # Try to find better
    for target_rank in [24, 23]:
        if time.time() - start_time > limit:
            break
        for attempt in range(1000):
            if time.time() - start_time > limit:
                break
            U = np.random.randint(-2, 3, (n*m, target_rank))
            V = np.random.randint(-2, 3, (m*p, target_rank))
            W = np.random.randint(-2, 3, (p*n, target_rank))
            
            # Simple check
            if check_solution(U, V, W, target_rank):
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
        if best_rank <= target_rank:
            break
            
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
