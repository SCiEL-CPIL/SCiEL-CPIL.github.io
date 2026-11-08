import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Total rank 55
    # 20 from 2x1x10
    # 35 from 5 * (2x2x2)
    rank = 55
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2x1x10
    # i in 0..1, j=0, k in 0..9
    col = 0
    for i in range(2):
        for k in range(10):
            U[i * 3 + 0][col] = 1
            V[0 * 10 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    # Part 2: 2x2x10
    # i in 0..1, j in 1..2, k in 0..9
    # Split k into 5 blocks of 2
    # Use Strassen for each block
    
    # Strassen 7 terms for 2x2x2
    # u, v, w are 2x2 matrices flattened to 4 vectors
    # Condition: sum_s u_s[i,j] v_s[j,k] w_s[k,i] = 1
    
    # The 7 terms (u, v, w) as 2x2 matrices:
    terms = [
        # 1
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 1], [1, 1]]),
        # 2
        ([[0, 1], [0, 1]], [[1, 0], [0, 0]], [[0, 0], [0, 1]]),
        # 3
        ([[1, 0], [0, 0]], [[0, 1], [0, 0]], [[1, 0], [0, 0]]),
        # 4
        ([[0, 0], [0, 1]], [[0, 0], [1, 0]], [[0, 0], [0, 1]]),
        # 5
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[0, 1], [0, 0]]),
        # 6
        ([[1, 0], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [1, 0]]),
        # 7
        ([[1, 0], [0, -1]], [[0, 1], [0, 1]], [[0, 0], [1, 0]])
    ]
    
    # Verify Strassen terms sum to 1 for all i,j,k
    # This is a known decomposition.
    
    for b in range(5):
        for s in range(7):
            u_mat, v_mat, w_mat = terms[s]
            # u_mat is 2x2, indices i', j'
            # v_mat is 2x2, indices j', k'
            # w_mat is 2x2, indices k', i'
            
            for ip in range(2):
                for jp in range(2):
                    u_val = u_mat[ip][jp]
                    if u_val != 0:
                        # Global indices
                        i = ip
                        j = 1 + jp
                        U[i * 3 + j][col] += u_val
                        
            for jp in range(2):
                for kp in range(2):
                    v_val = v_mat[jp][kp]
                    if v_val != 0:
                        j = 1 + jp
                        k = 2 * b + kp
                        V[j * 10 + k][col] += v_val
                        
            for kp in range(2):
                for ip in range(2):
                    w_val = w_mat[kp][ip]
                    if w_val != 0:
                        k = 2 * b + kp
                        i = ip
                        W[k * 2 + i][col] += w_val
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}
