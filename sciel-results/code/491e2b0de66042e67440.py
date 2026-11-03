import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    def schoolbook():
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
    
    def verify(U, V, W):
        rank = len(U[0])
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Define 23-rank 3x3x3 decomposition terms
    # Each term: (u_9, v_9, w_9) where u,v,w are flattened 3x3 matrices
    # u[i*3+j] for A[i,j], v[j*3+k] for B[j,k], w[k*3+i] for C[k,i]
    
    # Standard 23-rank decomposition of 3x3x3 matrix multiplication
    terms_23 = [
        # Term 1
        ([1,0,0, 0,0,0, 0,0,0], [1,0, 0,0, 0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 2
        ([0,1,0, 0,0,0, 0,0,0], [0,0, 1,0, 0,0], [0,1,0, 0,0,0, 0,0,0]),
        # Term 3
        ([0,0,0, 1,0,0, 0,0,0], [0,0, 0,1, 0,0], [0,0,1, 0,0,0, 0,0,0]),
        # Term 4
        ([0,0,0, 0,1,0, 0,0,0], [0,0, 0,0, 1,0], [0,0,0, 1,0,0, 0,0,0]),
        # Term 5
        ([0,0,0, 0,0,0, 1,0,0], [0,0, 0,0, 0,1], [0,0,0, 0,1,0, 0,0,0]),
        # Term 6
        ([0,0,0, 0,0,1, 0,0,0], [0,0, 0,0, 0,0], [0,0,0, 0,0,1, 0,0,0]),
        # This is not correct - I'm just guessing.
    ]
    
    # I cannot reliably construct the 23 terms. Return schoolbook.
    return schoolbook()

# EVOLVE-BLOCK-END
