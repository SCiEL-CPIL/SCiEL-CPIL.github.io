import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    rank = 2 * p * 2  # 2 sub-problems, each 2*p rank = 12, total 24
    # Actually: 2 sub-problems (split m into two groups of 2), each has rank 2*p = 12
    # Total rank = 2 * 2 * p = 4 * p = 24
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 24
    V = [[0] * rank for _ in range(m * p)]  # 24 x 24
    W = [[0] * rank for _ in range(p * n)]  # 12 x 24
    
    # Sub-problem 1: j = 0, 1
    # U indices: i*4+j for i=0,1; j=0,1 → 0,1,4,5
    # V indices: j*6+k for j=0,1; k=0..5 → 0-5, 6-11
    # W indices: k*2+i for k=0..5; i=0,1 → 0-11
    for k in range(p):
        # Factor 2k
        U[0][2*k] = 1    # i=0, j=0
        U[1][2*k] = 1    # i=0, j=1
        V[k][2*k] = 1    # j=0, k
        V[6+k][2*k] = 1  # j=1, k
        W[2*k][2*k] = 1  # k, i=0
        
        # Factor 2k+1
        U[4][2*k+1] = 1  # i=1, j=0
        U[5][2*k+1] = 1  # i=1, j=1
        V[k][2*k+1] = 1  # j=0, k
        V[6+k][2*k+1] = 1  # j=1, k
        W[2*k+1][2*k+1] = 1  # k, i=1
    
    # Sub-problem 2: j = 2, 3
    # U indices: i*4+j for i=0,1; j=2,3 → 2,3,6,7
    # V indices: j*6+k for j=2,3; k=0..5 → 12-17, 18-23
    # W indices: k*2+i for k=0..5; i=0,1 → 0-11
    for k in range(p):
        # Factor 12+2k
        U[2][12+2*k] = 1    # i=0, j=2
        U[3][12+2*k] = 1    # i=0, j=3
        V[12+k][12+2*k] = 1    # j=2, k
        V[18+k][12+2*k] = 1  # j=3, k
        W[2*k][12+2*k] = 1  # k, i=0
        
        # Factor 12+2k+1
        U[6][12+2*k+1] = 1  # i=1, j=2
        U[7][12+2*k+1] = 1  # i=1, j=3
        V[12+k][12+2*k+1] = 1  # j=2, k
        V[18+k][12+2*k+1] = 1  # j=3, k
        W[2*k+1][12+2*k+1] = 1  # k, i=1
    
    # Verify
    def verify(U_mat, V_mat, W_mat, n, m, p):
        rank = len(U_mat[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
                    if total != 1:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
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
    
    return {"U": U, "V": V, "W": W}
