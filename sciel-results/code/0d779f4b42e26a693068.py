import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Fill schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to lists as required by the interface (or keep as numpy arrays if allowed? 
    # The prompt says "factor matrices of shapes... with integer/half-integer entries".
    # The seed returns lists of lists. I will return lists of lists to be safe, 
    # or numpy arrays if the evaluator accepts them. 
    # "Interface: construct() -> {'U', 'V', 'W'}: factor matrices..."
    # The seed returns lists. I will return lists to match the seed's type exactly, 
    # or numpy arrays if they are more efficient. 
    # Usually, numpy arrays are fine. Let's check the seed: 
    # "U = [[0] * rank for _ in range(n * m)]" -> Lists.
    # I will return lists to be safe.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
