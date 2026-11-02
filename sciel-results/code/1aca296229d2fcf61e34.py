import numpy as np
import time

DIMENSIONS = [2, 4, 5]
DEADLINE = 110.0  # Safety margin before 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank = n * m * p
    
    # Initialize matrices
    # Shapes: (n*m, rank), (m*p, rank), (p*n, rank)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
                # Deadline check
                if time.time() - start_time > DEADLINE:
                    break
            if time.time() - start_time > DEADLINE:
                break
        if time.time() - start_time > DEADLINE:
            break
            
    # Convert to lists of lists as per interface requirement?
    # The interface says: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries"
    # The seed returns lists of lists. I should probably return lists of lists to be safe, or NumPy arrays if the evaluator accepts them.
    # The seed uses lists. I will use lists to match the seed exactly.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
