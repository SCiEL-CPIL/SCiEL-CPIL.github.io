import time
import sys

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    # Strassen vectors for 2x2x2
    # u: (i*m+j) -> 4
    # v: (j*p_loc+k_loc) -> 4 (p_loc=2)
    # w: (k_loc*n+i) -> 4 (n=2)
    
    u_vectors = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, -1, 0]
    ]
    v_vectors = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_vectors = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block in range(4):
        k_offset = block * 2
        for r in range(7):
            # U is same for all blocks
            for i in range(n):
                for j in range(m):
                    U[i*m + j][col] = u_vectors[r][i*m + j]
            
            # V shifted by k_offset
            for j in range(m):
                for k_loc in range(2):
                    V[j*p + k_offset + k_loc][col] = v_vectors[r][j*2 + k_loc]
            
            # W shifted by k_offset
            for k_loc in range(2):
                for i in range(n):
                    W[(k_offset + k_loc)*n + i][col] = w_vectors[r][k_loc*2 + i]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
