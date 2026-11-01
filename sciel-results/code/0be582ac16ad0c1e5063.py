"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return Strassen's rank-7 tensor decomposition for 2x2 matrix multiplication.
    
    Convention:
      A is indexed as row-major: A_idx = i*m + j  (i=0,1; j=0,1)
      B is indexed as row-major: B_idx = j*p + k  (j=0,1; k=0,1)
      C (output) follows the trace convention W[k*n + i, r]:
        C_idx = k*n + i  (k=0,1; i=0,1)
      So C_00 -> idx 0, C_10 -> idx 1, C_01 -> idx 2, C_11 -> idx 3
    
    Strassen's 7 products:
      M1 = (A00 + A11)(B00 + B11)
      M2 = (A10 + A11)(B00)
      M3 = (A00)(B01 - B11)
      M4 = (A11)(B10 - B00)
      M5 = (A00 + A01)(B11)
      M6 = (A10 - A00)(B00 + B01)
      M7 = (A01 - A11)(B10 + B11)
    
    Reconstruction:
      C00 = M1 + M4 - M5 + M7
      C10 = M2 + M4
      C01 = M3 + M5
      C11 = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # M1 = (A00 + A11)(B00 + B11)
    # U: A00 (idx 0) and A11 (idx 3)
    U[0][0] = 1
    U[3][0] = 1
    # V: B00 (idx 0) and B11 (idx 3)
    V[0][0] = 1
    V[3][0] = 1
    # W: C00 (idx 0) and C11 (idx 3)
    W[0][0] = 1
    W[3][0] = 1
    
    # M2 = (A10 + A11)(B00)
    # U: A10 (idx 2) and A11 (idx 3)
    U[2][1] = 1
    U[3][1] = 1
    # V: B00 (idx 0)
    V[0][1] = 1
    # W: C10 (idx 1) with +1, C11 (idx 3) with -1
    W[1][1] = 1
    W[3][1] = -1
    
    # M3 = (A00)(B01 - B11)
    # U: A00 (idx 0)
    U[0][2] = 1
    # V: B01 (idx 1) with +1, B11 (idx 3) with -1
    V[1][2] = 1
    V[3][2] = -1
    # W: C01 (idx 2) with +1
    W[2][2] = 1
    
    # M4 = (A11)(B10 - B00)
    # U: A11 (idx 3)
    U[3][3] = 1
    # V: B10 (idx 2) with +1, B00 (idx 0) with -1
    V[2][3] = 1
    V[0][3] = -1
    # W: C00 (idx 0) with +1, C10 (idx 1) with +1
    W[0][3] = 1
    W[1][3] = 1
    
    # M5 = (A00 + A01)(B11)
    # U: A00 (idx 0) and A01 (idx 1)
    U[0][4] = 1
    U[1][4] = 1
    # V: B11 (idx 3)
    V[3][4] = 1
    # W: C00 (idx 0) with -1, C01 (idx 2) with +1
    W[0][4] = -1
    W[2][4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    # U: A10 (idx 2) with +1, A00 (idx 0) with -1
    U[2][5] = 1
    U[0][5] = -1
    # V: B00 (idx 0) and B01 (idx 1)
    V[0][5] = 1
    V[1][5] = 1
    # W: C11 (idx 3) with +1
    W[3][5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    # U: A01 (idx 1) with +1, A11 (idx 3) with -1
    U[1][6] = 1
    U[3][6] = -1
    # V: B10 (idx 2) and B11 (idx 3)
    V[2][6] = 1
    V[3][6] = 1
    # W: C00 (idx 0) with +1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

