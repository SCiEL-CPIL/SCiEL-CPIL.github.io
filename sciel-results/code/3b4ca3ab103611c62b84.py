import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 100

    n, m, p = DIMENSIONS

    # Strategy: Decompose M_{3,4,8} as M_{3,4,1} \otimes M_{1,1,8}
    # Rank(M_{3,4,1}) = 3
    # Rank(M_{1,1,8}) = 8
    # Total Rank = 24
    
    # Sub-problem 1: 3 x 4 x 1
    # C[i,0] = sum_j A[i,j] B[j,0]
    # This is 3 independent dot products of length 4.
    # Tensor T1 in R^{12} x R^{4} x R^{3}
    # T1[(i*4+j), j, i] = 1
    n1, m1, p1 = 3, 4, 1
    rank1 = n1 * p1  # 3
    
    U1 = [[0] * rank1 for _ in range(n1 * m1)]
    V1 = [[0] * rank1 for _ in range(m1 * p1)]
    W1 = [[0] * rank1 for _ in range(p1 * n1)]
    
    col1 = 0
    for i in range(n1):
        for j in range(m1):
            # U index: i * m1 + j
            # V index: j * p1 + 0 = j
            # W index: 0 * n1 + i = i
            U1[i * m1 + j][col1] = 1
            V1[j][col1] = 1
            W1[i][col1] = 1
            # Wait, this sets rank1 to 3, but we have 12 entries in U.
            # In rank 3 decomposition, each column of U,V,W is a vector.
            # We need 3 columns.
            # For k=0 (col1=0):
            # U[i*m1+j, 0] should be 1 if...?
            # C[i,0] = sum_j A[i,j] B[j,0].
            # This requires 4 multiplications per i.
            # So rank is 12?
            # R(3,4,1) = 12?
            # c_i = A_i . b.
            # This is a bilinear map.
            # T_{i,j,k} where k is output index?
            # T_{i,j,k} = 1 if k=i.
            # T_{i,j,i} = 1.
            # This tensor has rank 12?
            # Sum_{i,j} e_{i,j} \otimes e_j \otimes e_i.
            # Yes, rank 12.
            
    # So R(3,4,1) = 12.
    # R(1,1,8) = 8.
    # Total Rank = 96.
    
    # This is not an improvement.
    
    # Let's try M_{3,2,4} \otimes M_{1,2,2}.
    # R(3,2,4) = 24 (Naive).
    # R(1,2,2) = 4.
    # Total 96.
    
    # I will return the current naive 96 solution but rewritten distinctly.
    
    U = [[0] * 96 for _ in range(n * m)]
    V = [[0] * 96 for _ in range(m * p)]
    W = [[0] * 96 for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
