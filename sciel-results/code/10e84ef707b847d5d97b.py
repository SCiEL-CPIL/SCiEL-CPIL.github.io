"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-7 exact tensor decomposition for 2x2 matrix multiplication (Strassen).
    
    This implements Strassen's algorithm, which is rank-minimal for 2x2
    matrix multiplication over the reals (proven lower bound is 7).
    
    Indexing convention:
      U has shape (n*m, rank) = (4, 7); row index = i*m + j  (A[i,j])
      V has shape (m*p, rank) = (4, 7); row index = j*p + k  (B[j,k])
      W has shape (p*n, rank) = (4, 7); row index = k*n + i  (C[i,k], trace convention)
    
    The tensor identity: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (U rows): i*m + j
    # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    
    # Indices for B (V rows): j*p + k
    # B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    
    # Indices for C (W rows): k*n + i
    # C[0,0] (i=0,k=0) -> 0*2+0 = 0
    # C[0,1] (i=0,k=1) -> 1*2+0 = 2
    # C[1,0] (i=1,k=0) -> 0*2+1 = 1
    # C[1,1] (i=1,k=1) -> 1*2+1 = 3
    
    # Strassen's 7 intermediate products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    
    # Output reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Column 0: M1
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11
    
    # Column 1: M2
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    W[1][1] = 1  # C10
    W[3][1] = -1 # C11
    
    # Column 2: M3
    U[0][2] = 1  # A00
    V[1][2] = 1  # B01
    V[3][2] = -1 # B11
    W[2][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # Column 3: M4
    U[3][3] = 1  # A11
    V[2][3] = 1  # B10
    V[0][3] = -1 # B00
    W[0][3] = 1  # C00
    W[1][3] = 1  # C10
    
    # Column 4: M5
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    W[0][4] = -1 # C00
    W[2][4] = 1  # C01
    
    # Column 5: M6
    U[2][5] = 1  # A10
    U[0][5] = -1 # A00
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    W[3][5] = 1  # C11
    
    # Column 6: M7
    U[1][6] = 1  # A01
    U[3][6] = -1 # A11
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    W[0][6] = 1  # C00
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
