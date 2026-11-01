"""
Strassen's rank-7 tensor decomposition for 2x2 matrix multiplication.
Verified via the tensor identity:
  sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i, k} (in the appropriate sense)

The decomposition is known to be rank-minimal (rank 7 is optimal for 2x2 matrix mult).
"""
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return Strassen's rank-7 tensor decomposition for 2x2 matrix multiplication.
    
    Shapes:
      U: (n*m, rank) = (4, 7), row index i*m + j
      V: (m*p, rank) = (4, 7), row index j*p + k
      W: (p*n, rank) = (4, 7), row index k*n + i  (official trace convention)
    
    The multiplication tensor satisfies:
      (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    Rank-r decomposition:
      T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Strassen's 7 intermediate products:
      M1 = (A00 + A11)(B00 + B11)
      M2 = (A10 + A11) * B00
      M3 = A00 * (B01 - B11)
      M4 = A11 * (B10 - B00)
      M5 = (A00 + A01) * B11
      M6 = (A10 - A00) * (B00 + B01)
      M7 = (A01 - A11) * (B10 + B11)
    
    Output:
      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    # U: shape (4, 7), row = i*m + j
    #   A00: row 0, A01: row 1, A10: row 2, A11: row 3
    U = [[0] * rank for _ in range(n * m)]
    
    # V: shape (4, 7), row = j*p + k
    #   B00: row 0, B01: row 1, B10: row 2, B11: row 3
    V = [[0] * rank for _ in range(m * p)]
    
    # W: shape (4, 7), row = k*n + i
    #   C00: k=0,i=0 -> row 0
    #   C10: k=0,i=1 -> row 1
    #   C01: k=1,i=0 -> row 2
    #   C11: k=1,i=1 -> row 3
    W = [[0] * rank for _ in range(p * n)]
    
    # r=0: M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1   # A00
    U[3][0] = 1   # A11
    V[0][0] = 1   # B00
    V[3][0] = 1   # B11
    W[0][0] = 1   # C00
    W[3][0] = 1   # C11
    
    # r=1: M2 = (A10 + A11) * B00
    U[2][1] = 1   # A10
    U[3][1] = 1   # A11
    V[0][1] = 1   # B00
    W[1][1] = 1   # C10
    W[3][1] = -1  # C11
    
    # r=2: M3 = A00 * (B01 - B11)
    U[0][2] = 1   # A00
    V[1][2] = 1   # B01
    V[3][2] = -1  # B11
    W[2][2] = 1   # C01
    W[3][2] = 1   # C11
    
    # r=3: M4 = A11 * (B10 - B00)
    U[3][3] = 1   # A11
    V[2][3] = 1   # B10
    V[0][3] = -1  # B00
    W[0][3] = 1   # C00
    W[1][3] = 1   # C10
    
    # r=4: M5 = (A00 + A01) * B11
    U[0][4] = 1   # A00
    U[1][4] = 1   # A01
    V[3][4] = 1   # B11
    W[0][4] = -1  # C00
    W[2][4] = 1   # C01
    
    # r=5: M6 = (A10 - A00) * (B00 + B01)
    U[2][5] = 1   # A10
    U[0][5] = -1  # A00
    V[0][5] = 1   # B00
    V[1][5] = 1   # B01
    W[3][5] = 1   # C11
    
    # r=6: M7 = (A01 - A11) * (B10 + B11)
    U[1][6] = 1   # A01
    U[3][6] = -1  # A11
    V[2][6] = 1   # B10
    V[3][6] = 1   # B11
    W[0][6] = 1   # C00
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
