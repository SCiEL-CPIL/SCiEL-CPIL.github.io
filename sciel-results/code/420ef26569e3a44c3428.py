"""Rewritten program for 2x2 matrix multiplication rank minimization."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    n, m, p = 2, 2, 2
    
    # Strassen's algorithm for 2x2 matrix multiplication uses 7 multiplications.
    # It is known that the tensor rank of 2x2 matrix multiplication is 7.
    # Therefore, rank 7 is optimal.
    
    rank = 7
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # shape (n*m, rank) = (4, 7)
    V = [[0] * rank for _ in range(m * p)]  # shape (m*p, rank) = (4, 7)
    W = [[0] * rank for _ in range(p * n)]  # shape (p*n, rank) = (4, 7)
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Indices for U: i*m + a (i is row of A, a is col of A)
    # Indices for V: a*p + k (a is row of B, k is col of B)
    # Indices for W: k*n + i (k is col of C, i is row of C)
    
    # M1: A00 + A11 times B00 + B11
    U[0][0] = 1   # A00
    U[3][0] = 1   # A11
    V[0][0] = 1   # B00
    V[3][0] = 1   # B11
    
    # M2: (A10 + A11) times B00
    U[2][1] = 1   # A10
    U[3][1] = 1   # A11
    V[0][1] = 1   # B00
    
    # M3: A00 times (B01 - B11)
    U[0][2] = 1   # A00
    V[1][2] = 1   # B01
    V[3][2] = -1  # -B11
    
    # M4: A11 times (B10 - B00)
    U[3][3] = 1   # A11
    V[2][3] = 1   # B10
    V[0][3] = -1  # -B00
    
    # M5: (A00 + A01) times B11
    U[0][4] = 1   # A00
    U[1][4] = 1   # A01
    V[3][4] = 1   # B11
    
    # M6: (A10 - A00) times (B00 + B01)
    U[2][5] = 1   # A10
    U[0][5] = -1  # -A00
    V[0][5] = 1   # B00
    V[1][5] = 1   # B01
    
    # M7: (A01 - A11) times (B10 + B11)
    U[1][6] = 1   # A01
    U[3][6] = -1  # -A11
    V[2][6] = 1   # B10
    V[3][6] = 1   # B11
    
    # W coefficients for C[i,k]
    # C[0,0] = M1 + M4 - M5 + M7
    W[0][0] = 1   # M1
    W[0][3] = 1   # M4
    W[0][4] = -1  # -M5
    W[0][6] = 1   # M7
    
    # C[0,1] = M3 + M5
    W[2][2] = 1   # M3
    W[2][4] = 1   # M5
    
    # C[1,0] = M2 + M4
    W[1][1] = 1   # M2
    W[1][3] = 1   # M4
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3][0] = 1   # M1
    W[3][1] = -1  # -M2
    W[3][2] = 1   # M3
    W[3][5] = 1   # M6
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
