import numpy as np
import random
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = 17
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = get_schoolbook()
    
    # Try to find rank 17 solution
    for _ in range(10000):
        if time.time() > deadline:
            break
            
        # Random initialization of U and V with small integers
        # We want sums to be 1.
        U = np.zeros((n * m, target_rank), dtype=int)
        V = np.zeros((m * p, target_rank), dtype=int)
        
        # Fill with random -1, 0, 1
        for r in range(target_rank):
            # Make sure each column has some activity
            for _ in range(3):
                ui = random.randint(0, n * m - 1)
                U[ui, r] = random.choice([-1, 1])
            for _ in range(3):
                vj = random.randint(0, m * p - 1)
                V[vj, r] = random.choice([-1, 1])
                
        # Check condition: for all k, i, exists r such that sum_j U[i*m+j, r] V[j*p+k, r] == 1
        valid = True
        W = np.zeros((p * n, target_rank), dtype=int)
        
        for k in range(p):
            for i in range(n):
                found = False
                for r in range(target_rank):
                    # Calculate sum over j
                    s = 0
                    for j in range(m):
                        s += U[i * m + j, r] * V[j * p + k, r]
                    if s == 1:
                        W[k * n + i, r] = 1
                        found = True
                        break
                if not found:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            # Convert to lists of ints
            res = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            return res
            
    return best_result
# EVOLVE-BLOCK-END
