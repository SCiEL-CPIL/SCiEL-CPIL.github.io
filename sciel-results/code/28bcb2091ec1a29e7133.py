import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    # Rank = n * m * p
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i, j] corresponds to row i*m + j
                U[i * m + j, col] = 1.0
                # V[j, k] corresponds to row j*p + k
                V[j * p + k, col] = 1.0
                # W[k, i] corresponds to row k*n + i
                W[k * n + i, col] = 1.0
                col += 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # Convert to lists as per interface requirement if needed, 
    # but the prompt says "factor matrices of shapes...". 
    # The seed returns lists of lists. 
    # I will return lists of lists to be safe with the "integer/half-integer" 
    # check if the evaluator expects exact types, though numpy floats 
    # 1.0 are exactly 1.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
