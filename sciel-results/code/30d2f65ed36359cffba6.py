# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-39 decomposition for 2x2 x 2x13 matrix multiplication.
    
    For each column k (0..12), the multiplication of a 2x2 matrix A by a 2x1 vector b = (b0, b1)^T
    can be done with rank 3 using the formula:
      c0 = (A00 + A01 + A11)(b0 + b1) - A01*b0 - A11*b1
      c1 = (A10 + A11)(b0 + b1) - A11*b1
    
    This gives 3 rank-1 terms per column:
      Term 1: (A00 + A01 + A11) * (b0 + b1) -> contributes +1 to c0, +1 to c1
      Term 2: A01 * b0 -> contributes -1 to c0
      Term 3: A11 * b1 -> contributes -1 to c0, -1 to c1
    
    Total rank = 13 * 3 = 39.
    """
    n, m, p = 2, 2, 13
    rank = p * 3  # 39
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 39)
    V = [[0] * rank for _ in range(m * p)]  # shape (26, 39)
    W = [[0] * rank for _ in range(p * n)]  # shape (26, 39)
    
    col = 0
    for k in range(p):
        # Term 1: (A00 + A01 + A11) * (b0 + b1)
        # U: A00, A01, A11 each get coefficient 1
        U[0][col] = 1   # A[0,0]
        U[1][col] = 1   # A[0,1]
        U[3][col] = 1   # A[1,1]
        # V: b0, b1 each get coefficient 1
        V[0 * p + k][col] = 1   # B[0, k]
        V[1 * p + k][col] = 1   # B[1, k]
        # W: contributes +1 to c0, +1 to c1
        W[k * n + 0][col] = 1   # C[0, k]
        W[k * n + 1][col] = 1   # C[1, k]
        col += 1
        
        # Term 2: A01 * b0
        # U: A01 gets coefficient 1
        U[1][col] = 1   # A[0,1]
        # V: b0 gets coefficient 1
        V[0 * p + k][col] = 1   # B[0, k]
        # W: contributes -1 to c0
        W[k * n + 0][col] = -1  # C[0, k]
        col += 1
        
        # Term 3: A11 * b1
        # U: A11 gets coefficient 1
        U[3][col] = 1   # A[1,1]
        # V: b1 gets coefficient 1
        V[1 * p + k][col] = 1   # B[1, k]
        # W: contributes -1 to c0, -1 to c1
        W[k * n + 0][col] = -1  # C[0, k]
        W[k * n + 1][col] = -1  # C[1, k]
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
