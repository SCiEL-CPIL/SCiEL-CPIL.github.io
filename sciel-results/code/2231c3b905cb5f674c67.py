# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = 3, 3, 7
    
    if _time_left() < 5:
        return _schoolbook(n, m, p)
    
    # Split into 3x3, 3x3, 3x1
    # Block 1: cols 0-2
    # Block 2: cols 3-5
    # Block 3: col 6
    
    # Rank 27 for 3x3 * 3x3 (Schoolbook)
    # Rank 9 for 3x3 * 3x1 (Schoolbook)
    
    # Total rank: 27 + 9 = 36 (sharing U for the two 3x3 blocks)
    
    U = []
    V = []
    W = []
    
    # Part 1: Two 3x3 blocks
    # Schoolbook for 3x3 * 3x3
    # U[i,j] = 1, V[j,k] = 1, W[k,i] = 1
    # We iterate over i, j, k
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = [0]*9
                u[i*3+j] = 1
                
                # V is 21x1 (3 blocks of 3)
                v = [0]*21
                # Block 1 (cols 0-2): index j*3+k in the 3x3 B matrix
                # B is 3x7. B[j, 0..2] is block 1.
                # v index for block 1: j*3 + k
                v[j*3 + k] = 1
                # Block 2 (cols 3-5): B[j, 3..5]
                # v index for block 2: 9 + j*3 + k
                v[9 + j*3 + k] = 1
                # Block 3 (col 6): 0
                # v[18 + j] = 0
                
                # W is 21x1 (3 blocks of 3)
                w = [0]*21
                # C is 3x7. C[i, 0..2] is block 1.
                # W index for block 1: k*3 + i
                w[k*3 + i] = 1
                # Block 2 (cols 3-5): C[i, 3..5]
                # W index for block 2: 9 + k*3 + i
                w[9 + k*3 + i] = 1
                # Block 3 (col 6): 0
                
                U.append(u)
                V.append(v)
                W.append(w)
                
    # Part 2: One 3x1 block
    # Schoolbook for 3x3 * 3x1
    # C[i, 6] = sum_j A[i,j] B[j, 6]
    for i in range(3):
        for j in range(3):
            u = [0]*9
            u[i*3+j] = 1
            
            v = [0]*21
            # Block 3 (col 6): B[j, 6]
            # v index for block 3: 18 + j
            v[18 + j] = 1
            
            w = [0]*21
            # Block 3 (col 6): C[i, 6]
            # W index for block 3: 18 + i
            w[18 + i] = 1
            
            U.append(u)
            V.append(v)
            W.append(w)
            
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
    rank = n * m * p
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
# EVOLVE-BLOCK-END
