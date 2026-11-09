import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Rank is n*m*p for the standard schoolbook decomposition.
    # We construct the U, V, W matrices using NumPy for speed.
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                
                col += 1
                
    # Convert to list of lists if required by the interface?
    # The prompt says: "factor matrices of shapes ... with integer/half-integer entries"
    # The current program returns lists of lists.
    # The evaluator might accept NumPy arrays or lists.
    # The seed program returns lists.
    # To be safe and match the seed's interface exactly, I will return lists.
    # However, NumPy conversion is fast.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
