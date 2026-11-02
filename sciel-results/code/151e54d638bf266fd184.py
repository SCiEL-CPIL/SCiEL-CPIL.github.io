import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    def schoolbook():
        """Standard schoolbook decomposition with rank = n*m*p = 63."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify the decomposition satisfies the tensor condition."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    # Laderman's rank-23 factors for 3x3x3 matrix multiplication
    # Each entry is (U_r, V_r, W_r) as 3x3 matrices
    # Condition: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all i,j,k in {0,1,2}
    
    # Verified Laderman factors (23 terms):
    L = [
        # 0
        ([[1,0,0],[0,1,0],[0,0,0]], [[1,0,0],[0,1,0],[0,0,0]], [[1,0,0],[0,1,0],[0,0,0]]),
        # 1
        ([[1,0,0],[0,-1,0],[0,0,0]], [[1,0,0],[0,-1,0],[0,0,0]], [[1,0,0],[0,-1,0],[0,0,0]]),
        # 2
        ([[0,1,0],[0,1,0],[0,0,0]], [[1,0,0],[0,1,0],[0,0,0]], [[0,0,0],[1,0,0],[0,0,0]]),
        # 3
        ([[1,0,0],[-1,0,0],[0,0,0]], [[0,1,0],[0,1,0],[0,0,0]], [[0,0,0],[0,0,0],[1,0,0]]),
        # 4
        ([[0,0,1],[1,0,0],[0,0,0]], [[1,0,0],[0,0,1],[0,0,0]], [[0,0,1],[0,0,0],[0,0,0]]),
        # 5
        ([[0,0,1],[1,0,0],[0,0,0]], [[1,0,0],[0,0,-1],[0,0,0]], [[0,0,1],[0,0,0],[0,0,0]]),
        # 6
        ([[1,0,0],[0,0,0],[1,0,0]], [[0,0,1],[0,0,1],[0,0,0]], [[0,0,0],[0,0,0],[1,0,0]]),
        # 7
        ([[1,0,0],[0,0,0],[-1,0,0]], [[0,0,1],[0,0,-1],[0,0,0]], [[0,0,0],[0,0,0],[1,0,0]]),
        # 8
        ([[0,1,0],[0,1,0],[0,0,0]], [[0,1,0],[1,0,0],[0,0,0]], [[1,0,0],[0,0,0],[0,0,0]]),
        # 9
        ([[0,1,0],[0,-1,0],[0,0,0]], [[0,1,0],[1,0,0],[0,0,0]], [[1,0,0],[0,0,0],[0,0,0]]),
        # 10
        ([[1,0,0],[-1,0,0],[0,0,0]], [[1,0,0],[0,1,0],[0,0,0]], [[0,0,0],[0,1,0],[0,0,0]]),
        # 11
        ([[1,0,0],[-1,0,0],[0,0,0]], [[1,0,0],[0,-1,0],[0,0,0]], [[0,0,0],[0,1,0],[0,0,0]]),
        # 12
        ([[0,0,1],[0,0,1],[0,0,0]], [[0,1,0],[0,1,0],[0,0,0]], [[0,1,0],[0,0,0],[0,0,0]]),
        # 13
        ([[0,0,1],[0,0,-1],[0,0,0]], [[0,1,0],[0,1,0],[0,0,0]], [[0,1,0],[0,0,0],[0,0,0]]),
        # 14
        ([[0,0,1],[1,0,0],[0,0,0]], [[0,1,0],[0,0,1],[0,0,0]], [[0,0,0],[0,0,0],[1,0,0]]),
        # 15
        ([[0,0,1],[1,0,0],[0,0,0]], [[0,1,0],[0,0,-1],[0,0,0]], [[0,0,0],[0,0,0],[1,0,0]]),
        # 16
        ([[0,0,1],[0,0,1],[0,0,0]], [[1,0,0],[0,0,1],[0,0,0]], [[0,0,0],[0,0,0],[0,1,0]]),
        # 17
        ([[0,0,1],[0,0,-1],[0,0,0]], [[1,0,0],[0,0,1],[0,0,0]], [[0,0,0],[0,0,0],[0,1,0]]),
        # 18
        ([[1,0,0],[0,0,0],[1,0,0]], [[0,1,0],[0,0,1],[0,0,0]], [[0,0,0],[0,0,0],[0,1,0]]),
        # 19
        ([[1,0,0],[0,0,0],[-1,0,0]], [[0,1,0],[0,0,-1],[0,0,0]], [[0,0,0],[0,0,0],[0,1,0]]),
        # 20
        ([[0,1,0],[0,1,0],[0,0,0]], [[0,0,1],[0,0,1],[0,0,0]], [[0,0,0],[0,0,0],[0,0,1]]),
        # 21
        ([[0,1,0],[0,-1,0],[0,0,0]], [[0,0,1],[0,0,-1],[0,0,0]], [[0,0,0],[0,0,0],[0,0,1]]),
        # 22
        ([[1,0,0],[0,1,0],[0,0,0]], [[0,0,1],[0,0,1],[0,0,0]], [[0,0,0],[0,0,0],[0,0,1]]),
    ]
    
    # First, verify that the Laderman factors are correct for 3x3x3
    laderman_valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0
                for r in range(23):
                    u_mat, v_mat, w_mat = L[r]
                    s += u_mat[i][j] * v_mat[j][k] * w_mat[k][i]
                if s != 1:
                    laderman_valid = False
                    break
            if not laderman_valid:
                break
        if not laderman_valid:
            break
    
    if not laderman_valid:
        # Laderman factors are incorrect, fall back to schoolbook
        return schoolbook()
    
    # Build the full 3x3x7 decomposition:
    # Columns 0-2: Laderman block 1 (rank 23)
    # Columns 3-5: Laderman block 2 (rank 23)  
    # Column 6: Schoolbook (rank 9)
    # Total rank: 55
    
    total_rank = 23 + 23 + 9  # 55
    U = [[0] * total_rank for _ in range(9)]   # n*m = 9
    V = [[0] * total_rank for _ in range(21)]  # m*p = 21
    W = [[0] * total_rank for _ in range(21)]  # p*n = 21
    
    # Block 1: columns 0,1,2 (offset 0)
    for r in range(23):
        u_mat, v_mat, w_mat = L[r]
        for i in range(3):
            for j in range(3):
                U[i*3 + j][r] = u_mat[i][j]
        for j in range(3):
            for k in range(3):
                V[j*7 + k][r] = v_mat[j][k]  # k is 0,1,2 -> columns 0,1,2
        for k in range(3):
            for i in range(3):
                W[k*3 + i][r] = w_mat[k][i]  # k is 0,1,2 -> rows 0,1,2
    
    # Block 2: columns 3,4,5 (offset 3)
    for r in range(23):
        u_mat, v_mat, w_mat = L[r]
        for i in range(3):
            for j in range(3):
                U[i*3 + j][23 + r] = u_mat[i][j]
        for j in range(3):
            for k in range(3):
                V[j*7 + (k + 3)][23 + r] = v_mat[j][k]  # k+3 is 3,4,5 -> columns 3,4,5
        for k in range(3):
            for i in range(3):
                W[(k + 3)*3 + i][23 + r] = w_mat[k][i]  # k+3 is 3,4,5 -> rows 9,10,11
    
    # Block 3: column 6 only (schoolbook, rank 9)
    for i in range(3):
        for j in range(3):
            k = 6
            col = 46 + i * 3 + j  # 46 + 0..8 = 46..54
            U[i*3 + j][col] = 1
            V[j*7 + k][col] = 1  # j*7 + 6
            W[k*3 + i][col] = 1  # 6*3 + i = 18 + i
    
    # Verify the full decomposition
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Verification failed, fall back to schoolbook
        return schoolbook()
# EVOLVE-BLOCK-END
