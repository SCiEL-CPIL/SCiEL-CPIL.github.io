"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-39 exact tensor decomposition for 2x2 * 2x11 matrix multiplication.
    
    Strategy: For 5 pairs of columns in the 2x11 matrix B, use the Strassen algorithm
    (7 multiplications per pair). For the remaining 1 column, use schoolbook (4 mults).
    Total rank = 5*7 + 1*4 = 35 + 4 = 39.
    
    Matrix multiplication convention:
      A is 2x2: A[i,j] where i=0,1 (row), j=0,1 (col)
      B is 2x11: B[j,k] where j=0,1 (row), k=0..10 (col)
      C = A @ B is 2x11: C[i,k] = sum_j A[i,j]*B[j,k]
    
    Tensor indexing:
      U has shape (n*m, rank) = (4, rank): index by i*m+j = i*2+j
      V has shape (m*p, rank) = (22, rank): index by j*p+k = j*11+k
      W has shape (p*n, rank) = (22, rank): index by k*n+i = k*2+i
    """
    n, m, p = 2, 2, 11
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]   # 4 x 39
    V = [[0] * rank for _ in range(m * p)]   # 22 x 39
    W = [[0] * rank for _ in range(p * n)]   # 22 x 39
    
    col = 0
    
    # Strassen decomposition for 5 pairs of columns
    # For columns (2t, 2t+1) where t = 0, 1, 2, 3, 4
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # 7 Strassen products:
        # P1 = (A00 + A11) * (B0k0 + B1k1)
        # P2 = (A10 + A11) * B0k0
        # P3 = A00 * (B1k0 - B0k0)
        # P4 = A11 * (B1k1 - B0k1)
        # P5 = (A00 + A01) * B1k1
        # P6 = (A10 - A00) * (B0k0 + B0k1)
        # P7 = (A01 - A11) * (B1k0 + B1k1)
        
        # Output:
        # C[0,k0] = P1 + P4 - P5 + P7
        # C[0,k1] = P3 + P5
        # C[1,k0] = P2 + P6
        # C[1,k1] = P1 + P3 - P2 + P6
        
        # Column for P1
        U[0][col] = 1    # A00
        U[3][col] = 1    # A11
        V[0 * p + k0][col] = 1   # B[0,k0]
        V[1 * p + k1][col] = 1   # B[1,k1]
        W[k0 * n + 0][col] = 1   # C[0,k0]
        W[k1 * n + 1][col] = 1   # C[1,k1]
        col += 1
        
        # Column for P2
        U[2][col] = 1    # A10
        U[3][col] = 1    # A11
        V[0 * p + k0][col] = 1   # B[0,k0]
        W[k0 * n + 1][col] = 1   # C[1,k0]
        col += 1
        
        # Column for P3
        U[0][col] = 1    # A00
        V[1 * p + k0][col] = 1   # B[1,k0]
        V[0 * p + k0][col] = -1  # -B[0,k0]
        W[k1 * n + 0][col] = 1   # C[0,k1]
        W[k1 * n + 1][col] = 1   # C[1,k1]
        col += 1
        
        # Column for P4
        U[3][col] = 1    # A11
        V[1 * p + k1][col] = 1   # B[1,k1]
        V[0 * p + k1][col] = -1  # -B[0,k1]
        W[k0 * n + 0][col] = 1   # C[0,k0]
        col += 1
        
        # Column for P5
        U[0][col] = 1    # A00
        U[1][col] = 1    # A01
        V[1 * p + k1][col] = 1   # B[1,k1]
        W[k0 * n + 0][col] = -1  # C[0,k0]
        W[k1 * n + 0][col] = 1   # C[0,k1]
        col += 1
        
        # Column for P6
        U[2][col] = 1    # A10
        U[0][col] = -1   # -A00
        V[0 * p + k0][col] = 1   # B[0,k0]
        V[0 * p + k1][col] = 1   # B[0,k1]
        W[k0 * n + 1][col] = 1   # C[1,k0]
        W[k1 * n + 1][col] = 1   # C[1,k1]
        col += 1
        
        # Column for P7
        U[1][col] = 1    # A01
        U[3][col] = -1   # -A11
        V[1 * p + k0][col] = 1   # B[1,k0]
        V[1 * p + k1][col] = 1   # B[1,k1]
        W[k0 * n + 0][col] = 1   # C[0,k0]
        col += 1
    
    # Schoolbook for the remaining single column k=10
    k = 10
    # M1 = A00 * B[0,k] -> contributes to C[0,k]
    U[0][col] = 1
    V[0 * p + k][col] = 1
    W[k * n + 0][col] = 1
    col += 1
    
    # M2 = A01 * B[1,k] -> contributes to C[0,k]
    U[1][col] = 1
    V[1 * p + k][col] = 1
    W[k * n + 0][col] = 1
    col += 1
    
    # M3 = A10 * B[0,k] -> contributes to C[1,k]
    U[2][col] = 1
    V[0 * p + k][col] = 1
    W[k * n + 1][col] = 1
    col += 1
    
    # M4 = A11 * B[1,k] -> contributes to C[1,k]
    U[3][col] = 1
    V[1 * p + k][col] = 1
    W[k * n + 1][col] = 1
    col += 1
    
    assert col == rank, f"Expected rank {rank} but used {col} columns"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
