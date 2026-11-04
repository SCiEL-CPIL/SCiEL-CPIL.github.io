"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    # Schoolbook decomposition: rank = n*m*p = 40
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try random search for lower rank
    for rank in range(rank_school - 1, 30, -1):
        if time.time() > deadline - 5:
            break
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            U = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(p*n)]
            if verify(U, V, W, rank):
                best = {"U": U, "V": V, "W": W}
                break
    
    return best
# EVOLVE-BLOCK-END
