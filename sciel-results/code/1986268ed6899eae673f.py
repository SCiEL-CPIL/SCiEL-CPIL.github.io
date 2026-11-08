import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = rank_schoolbook
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
    
    best_result = schoolbook()
    best_rank = rank_schoolbook
    
    # Try to find a better decomposition
    # The tensor T has T[(i,j),(j',k),(k',i')] = 1 iff i=i', j=j', k=k'
    # We want T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    
    # For a rank-r decomposition, we need:
    # For all (i,j,k): 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # This is a system of n*m*p equations in r*(n*m + m*p + p*n) unknowns.
    
    # Let's try a random search for rank < 30
    for target_rank in range(25, 30):
        if time.time() > deadline:
            break
        
        for trial in range(100):
            if time.time() > deadline:
                break
            
            # Generate random U, V, W
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Check if the decomposition is valid
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(target_rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(val - 1) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Convert to list format
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if target_rank < best_rank:
                best_rank = target_rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
            
            if time.time() > deadline:
                break
    
    return best_result
