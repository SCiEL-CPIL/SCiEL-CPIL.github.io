"""Matrix multiplication 2,2,3 - rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def construct():
    """
    Construct a rank-11 decomposition for 2x2 by 2x3 matrix multiplication.
    
    Strategy: Use Strassen's algorithm (rank 7) for the 2x2 sub-problem 
    (multiplying 2x2 by 2x2), then add 4 schoolbook terms for the third column.
    
    C = A * B where A is 2x2, B is 2x3, C is 2x3.
    C[:, :2] computed via Strassen (rank 7), C[:, 2] via schoolbook (rank 4).
    Total rank: 11.
    """
    n, m, p = 2, 2, 3
    
    # Strassen's 7 multiplications for 2x2 * 2x2:
    # Let A = [[a00, a01], [a10, a11]], B2 = [[b00, b01], [b10, b11]]
    # M1 = a00 * (b00 + b11)
    # M2 = (a00 + a01) * b11
    # M3 = (a10 + a11) * b00
    # M4 = a11 * (b01 + b10)
    # M5 = (a00 + a10) * (b00 + b01 - b11 - b10)
    # M6 = (a01 - a10) * (b10 + b11)
    # M7 = (a00 - a01) * (b00 + b01)
    # C00 = M1 + M4 - M2 + M6
    # C01 = M2 + M5
    # C10 = M3 + M7
    # C11 = M1 - M5 + M3 + M4
    
    # For the full problem, B has columns [B2 | b_col2] where b_col2 = [b02, b12]^T
    # C[:, 2] = A * [b02, b12]^T
    # C[0,2] = a00*b02 + a01*b12
    # C[1,2] = a10*b02 + a11*b12
    
    # Total rank = 7 (Strassen) + 4 (schoolbook for col 2) = 11
    
    rank = 11
    U = [[0] * rank for _ in range(n * m)]  # 4 x 11
    V = [[0] * rank for _ in range(m * p)]  # 6 x 11
    W = [[0] * rank for _ in range(p * n)]  # 6 x 11
    
    # Indexing:
    # U row: i*m + j = i*2 + j for A[i,j]
    # V row: j*p + k = j*3 + k for B[j,k]
    # W row: k*n + i = k*2 + i for C[i,k] (trace convention W[k*n+i, r])
    
    # Strassen part (columns 0-6):
    # M1 = a00 * (b00 + b11)
    c = 0
    U[0*2+0][c] = 1  # a00
    V[0*3+0][c] = 1  # b00
    V[1*3+1][c] = 1  # b11
    W[0*2+0][c] = 1  # C[0,0]
    W[1*2+1][c] = 1  # C[1,1]
    c += 1
    
    # M2 = (a00 + a01) * b11
    U[0*2+0][c] = 1  # a00
    U[0*2+1][c] = 1  # a01
    V[1*3+1][c] = 1  # b11
    W[0*2+0][c] = -1  # C[0,0]
    W[0*2+1][c] = 1  # C[0,1]
    c += 1
    
    # M3 = (a10 + a11) * b00
    U[1*2+0][c] = 1  # a10
    U[1*2+1][c] = 1  # a11
    V[0*3+0][c] = 1  # b00
    W[1*2+0][c] = 1  # C[1,0]
    W[1*2+1][c] = 1  # C[1,1]
    c += 1
    
    # M4 = a11 * (b01 + b10)
    U[1*2+1][c] = 1  # a11
    V[0*3+1][c] = 1  # b01
    V[1*3+0][c] = 1  # b10
    W[0*2+0][c] = 1  # C[0,0]
    W[1*2+1][c] = 1  # C[1,1]
    c += 1
    
    # M5 = (a00 + a10) * (b00 + b01 - b11 - b10)
    U[0*2+0][c] = 1  # a00
    U[1*2+0][c] = 1  # a10
    V[0*3+0][c] = 1  # b00
    V[0*3+1][c] = 1  # b01
    V[1*3+1][c] = -1  # b11
    V[1*3+0][c] = -1  # b10
    W[0*2+1][c] = 1  # C[0,1]
    W[1*2+1][c] = -1  # C[1,1]
    c += 1
    
    # M6 = (a01 - a10) * (b10 + b11)
    U[0*2+1][c] = 1  # a01
    U[1*2+0][c] = -1  # a10
    V[1*3+0][c] = 1  # b10
    V[1*3+1][c] = 1  # b11
    W[0*2+0][c] = 1  # C[0,0]
    c += 1
    
    # M7 = (a00 - a01) * (b00 + b01)
    U[0*2+0][c] = 1  # a00
    U[0*2+1][c] = -1  # a01
    V[0*3+0][c] = 1  # b00
    V[0*3+1][c] = 1  # b01
    W[1*2+0][c] = 1  # C[1,0]
    c += 1
    
    # Schoolbook for column 2 (columns 7-10):
    # C[0,2] = a00*b02 + a01*b12
    # C[1,2] = a10*b02 + a11*b12
    
    # Term 8: a00 * b02
    U[0*2+0][c] = 1
    V[0*3+2][c] = 1
    W[2*2+0][c] = 1  # C[0,2]
    c += 1
    
    # Term 9: a01 * b12
    U[0*2+1][c] = 1
    V[1*3+2][c] = 1
    W[2*2+0][c] = 1  # C[0,2]
    c += 1
    
    # Term 10: a10 * b02
    U[1*2+0][c] = 1
    V[0*3+2][c] = 1
    W[2*2+1][c] = 1  # C[1,2]
    c += 1
    
    # Term 11: a11 * b12
    U[1*2+1][c] = 1
    V[1*3+2][c] = 1
    W[2*2+1][c] = 1  # C[1,2]
    c += 1
    
    return {"U": U, "V": V, "W": W}

