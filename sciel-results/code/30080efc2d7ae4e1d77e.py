"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    # This is the known optimal rank for 2x2 matmul.
    #
    # U has shape (n*m, rank) = (4, 7)
    # V has shape (m*p, rank) = (4, 7)
    # W has shape (p*n, rank) = (4, 7)
    #
    # The decomposition satisfies:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = (A*B)[i,k]
    #
    # Using the standard Strassen decomposition with the trace convention W[k*n+i, r]:
    n, m, p = DIMENSIONS
    rank = 7
    
    # U[i*m + j, c] - coefficients for input matrix A
    # V[j*p + k, c] - coefficients for input matrix B
    # W[k*n + i, c] - coefficients for output C
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U[i*m+j, c]: row index i*m+j corresponds to A[i][j]
    # V[j*p+k, c]: row index j*p+k corresponds to B[j][k]
    # W[k*n+i, c]: row index k*n+i corresponds to C[i][k]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Column 0: M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00 (k=0,i=0)
    W[3][0] = 1  # C11 (k=1,i=1)
    
    # Column 1: M2 = (A10 + A11)B00
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    W[2][1] = 1  # C10 (k=0,i=1)
    
    # Column 2: M3 = A00(B01 - B11)
    U[0][2] = 1  # A00
    V[1][2] = 1  # B01
    V[3][2] = -1 # B11
    W[1][2] = 1  # C01 (k=1,i=0)
    
    # Column 3: M4 = A11(B10 - B00)
    U[3][3] = 1  # A11
    V[2][3] = 1  # B10
    V[0][3] = -1 # B00
    W[0][3] = 1  # C00 (k=0,i=0)
    W[2][3] = 1  # C10 (k=0,i=1)
    
    # Column 4: M5 = (A00 + A01)B11
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    W[0][4] = -1 # C00 (k=0,i=0)
    W[1][4] = 1  # C01 (k=1,i=0)
    
    # Column 5: M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1  # A10
    U[0][5] = -1 # A00
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    W[3][5] = 1  # C11 (k=1,i=1)
    
    # Column 6: M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1  # A01
    U[3][6] = -1 # A11
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    W[0][6] = 1  # C00 (k=0,i=0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
