import numpy as np
import random
import time
import sys

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115
    
    # Try to find a low rank decomposition
    for r in range(14, 31):
        if time.time() > deadline - 5:
            break
        
        # More attempts for lower ranks
        max_attempts = 500 if r < 18 else 200
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with small integers: -1, 0, 1
            U = np.random.randint(-1, 2, size=(n*m, r))
            V = np.random.randint(-1, 2, size=(m*p, r))
            W = np.random.randint(-1, 2, size=(p*n, r))
            
            # Vectorized verification
            valid = True
            for i in range(n):
                for j in range(m):
                    u_row = U[i*m+j, :]  # shape (r,)
                    V_block = V[j*p:(j+1)*p, :]  # shape (p, r)
                    W_block = W[i::n, :]  # shape (p, r)
                    
                    # For each c: sum_k V_block[k,c] * W_block[k,c]
                    # Then sum_c u_row[c] * that
                    inner = (V_block * W_block).sum(axis=0)  # shape (r,)
                    val = np.dot(u_row, inner)
                    
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If no low rank found, return schoolbook
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
