"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists as per seed, or numpy arrays?
    # The seed uses lists of lists. The evaluator likely accepts numpy arrays too, 
    # but lists are safer if the interface expects specific types. 
    # However, numpy is faster to generate. 
    # The prompt says "integer/half-integer entries". 
    # I will use numpy for generation and convert to lists if necessary, 
    # or just return numpy arrays if the evaluator handles them.
    # The seed returns lists of lists. I will stick to that to be safe, 
    # but generate them efficiently.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][column] = 1
                # V index: j*p + k
                V[j * p + k][column] = 1
                # W index: k*n + i
                W[k * n + i][column] = 1
                column += 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
