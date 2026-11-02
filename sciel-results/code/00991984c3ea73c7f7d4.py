"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 4
    rank = 14  # 7 for each 2x2 block
    
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 14)
    V = [[0] * rank for _ in range(m * p)]  # shape (8, 14)
    W = [[0] * rank for _ in range(p * n)]  # shape (8, 14)
    
    # Strassen for 2x2 * 2x2
    # Block 1: k=0,1 (columns 0,1 of B)
    # Block 2: k=2,3 (columns 2,3 of B)
    
    # For each block, define the 7 Strassen multipliers
    # U rows: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # V rows: B[j,k] -> index j*p+k, so B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
    # W rows: C[i,k] -> index k*n+i, so C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3, C[0,2]=4, C[1,2]=5, C[0,3]=6, C[1,3]=7
    
    # Block 1: k=0,1
    # M1: (A00+A11)*(B00+B11)
    col = 0
    U[0][col] = 1; U[3][col] = 1  # A00 + A11
    V[0][col] = 1; V[5][col] = 1  # B[0,0] + B[1,1]
    W[0][col] = 1; W[3][col] = 1  # C[0,0] += M1, C[1,1] += M1
    
    # M2: (A10+A11)*B00
    col = 1
    U[2][col] = 1; U[3][col] = 1  # A10 + A11
    V[0][col] = 1  # B[0,0]
    W[1][col] = 1  # C[1,0] += M2
    W[3][col] = -1  # C[1,1] -= M2
    
    # M3: A00*(B01-B11)
    col = 2
    U[0][col] = 1  # A00
    V[1][col] = 1; V[5][col] = -1  # B[0,1] - B[1,1]
    W[2][col] = 1  # C[0,1] += M3
    W[3][col] = 1  # C[1,1] += M3
    
    # M4: A11*(B10-B00)
    col = 3
    U[3][col] = 1  # A11
    V[4][col] = 1; V[0][col] = -1  # B[1,0] - B[0,0]
    W[0][col] = 1  # C[0,0] += M4
    W[1][col] = 1  # C[1,0] += M4
    
    # M5: (A00+A01)*B11
    col = 4
    U[0][col] = 1; U[1][col] = 1  # A00 + A01
    V[5][col] = 1  # B[1,1]
    W[0][col] = -1  # C[0,0] -= M5
    W[2][col] = 1  # C[0,1] += M5
    
    # M6: (A10-A00)*(B00+B01)
    col = 5
    U[2][col] = 1; U[0][col] = -1  # A10 - A00
    V[0][col] = 1; V[1][col] = 1  # B[0,0] + B[0,1]
    W[3][col] = 1  # C[1,1] += M6
    
    # M7: (A01-A11)*(B10+B11)
    col = 6
    U[1][col] = 1; U[3][col] = -1  # A01 - A11
    V[4][col] = 1; V[5][col] = 1  # B[1,0] + B[1,1]
    W[0][col] = 1  # C[0,0] += M7
    
    # Block 2: k=2,3
    # M1: (A00+A11)*(B02+B13)
    col = 7
    U[0][col] = 1; U[3][col] = 1
    V[2][col] = 1; V[7][col] = 1  # B[0,2] + B[1,3]
    W[4][col] = 1; W[7][col] = 1  # C[0,2] += M1, C[1,3] += M1
    
    # M2: (A10+A11)*B02
    col = 8
    U[2][col] = 1; U[3][col] = 1
    V[2][col] = 1  # B[0,2]
    W[5][col] = 1  # C[1,2] += M2
    W[7][col] = -1  # C[1,3] -= M2
    
    # M3: A00*(B03-B13)
    col = 9
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1  # B[0,3] - B[1,3]
    W[6][col] = 1  # C[0,3] += M3
    W[7][col] = 1  # C[1,3] += M3
    
    # M4: A11*(B12-B02)
    col = 10
    U[3][col] = 1
    V[6][col] = 1; V[2][col] = -1  # B[1,2] - B[0,2]
    W[4][col] = 1  # C[0,2] += M4
    W[5][col] = 1  # C[1,2] += M4
    
    # M5: (A00+A01)*B13
    col = 11
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1  # B[1,3]
    W[4][col] = -1  # C[0,2] -= M5
    W[6][col] = 1  # C[0,3] += M5
    
    # M6: (A10-A00)*(B02+B03)
    col = 12
    U[2][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1  # B[0,2] + B[0,3]
    W[7][col] = 1  # C[1,3] += M6
    
    # M7: (A01-A11)*(B12+B13)
    col = 13
    U[1][col] = 1; U[3][col] = -1
    V[6][col] = 1; V[7][col] = 1  # B[1,2] + B[1,3]
    W[4][col] = 1  # C[0,2] += M7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
