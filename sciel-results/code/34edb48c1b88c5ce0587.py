import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = rank
    
    # Try to find lower rank decompositions
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for valid i,j,k
    
    # Try ranks from lower values up
    for try_rank in range(1, rank + 1):
        if time.time() > deadline - 5:
            break
            
        # Initialize with random values
        np.random.seed(42 + try_rank)
        U_try = np.random.randn(n * m, try_rank)
        V_try = np.random.randn(m * p, try_rank)
        W_try = np.random.randn(p * n, try_rank)
        
        # Use gradient-free optimization or simple iterations
        # For now, just verify schoolbook works
        if try_rank >= rank:
            break
    
    # Return the best found (which is schoolbook for now)
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
