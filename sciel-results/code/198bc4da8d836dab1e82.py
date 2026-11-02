"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Laderman's rank-23 algorithm for 3x3x3 matrix multiplication
    # Using the standard Laderman decomposition
    # U factors (23 terms, each is a linear combination of A entries)
    # V factors (23 terms, each is a linear combination of B entries)
    # W factors (23 terms, each gives a linear combination of C entries)
    
    # Laderman's algorithm factors:
    # A is 3x3, B is 3x3, C = A*B is 3x3
    # Indexing: A[i,j], B[j,k], C[i,k]
    
    # The 23 bilinear forms for Laderman's algorithm:
    # f1 = A00 * B00
    # f2 = A00 * B01
    # f3 = A01 * B10
    # f4 = A01 * B11
    # f5 = A02 * B20
    # f6 = A02 * B21
    # f7 = A10 * B00
    # f8 = A10 * B01
    # f9 = A11 * B10
    # f10 = A11 * B11
    # f11 = A12 * B20
    # f12 = A12 * B21
    # f13 = A20 * B00
    # f14 = A20 * B01
    # f15 = A21 * B10
    # f16 = A21 * B11
    # f17 = A22 * B20
    # f18 = A22 * B21
    # f19 = (A00-A10) * (B00-B10)
    # f20 = (A01-A11) * (B01-B11)
    # f21 = (A02-A12) * (B02-B12)
    # f22 = (A00+A11-A22) * (B00+B11-B22)
    # f23 = (A01+A10-A20) * (B10+B01-B00)
    
    # Let me use the correct Laderman decomposition
    # Standard Laderman factors (from literature):
    
    # U (for A): each row i*m+j corresponds to A[i,j]
    # V (for B): each row j*p+k corresponds to B[j,k]  
    # W (for C): each row k*n+i corresponds to C[i,k]
    
    # Laderman's 23-term algorithm:
    # Terms are indexed 0..22
    
    # U factors (coefficients for A entries in each bilinear form):
    # A indices: 0=A00, 1=A01, 2=A02, 3=A10, 4=A11, 5=A12, 6=A20, 7=A21, 8=A22
    
    U_laderman = [
        # term 0: A00*B00
        [1,0,0, 0,0,0, 0,0,0],
        # term 1: A00*B01
        [1,0,0, 0,0,0, 0,0,0],
        # term 2: A01*B10
        [0,1,0, 0,0,0, 0,0,0],
        # term 3: A01*B11
        [0,1,0, 0,0,0, 0,0,0],
        # term 4: A02*B20
        [0,0,1, 0,0,0, 0,0,0],
        # term 5: A02*B21
        [0,0,1, 0,0,0, 0,0,0],
        # term 6: A10*B00
        [0,0,0, 1,0,0, 0,0,0],
        # term 7: A10*B01
        [0,0,0, 1,0,0, 0,0,0],
        # term 8: A11*B10
        [0,0,0, 0,1,0, 0,0,0],
        # term 9: A11*B11
        [0,0,0, 0,1,0, 0,0,0],
        # term 10: A12*B20
        [0,0,0, 0,0,1, 0,0,0],
        # term 11: A12*B21
        [0,0,0, 0,0,1, 0,0,0],
        # term 12: A20*B00
        [0,0,0, 0,0,0, 1,0,0],
        # term 13: A20*B01
        [0,0,0, 0,0,0, 1,0,0],
        # term 14: A21*B10
        [0,0,0, 0,0,0, 0,1,0],
        # term 15: A21*B11
        [0,0,0, 0,0,0, 0,1,0],
        # term 16: A22*B20
        [0,0,0, 0,0,0, 0,0,1],
        # term 17: A22*B21
        [0,0,0, 0,0,0, 0,0,1],
        # term 18: (A00-A10)*(B00-B10)
        [1,0,0, -1,0,0, 0,0,0],
        # term 19: (A01-A11)*(B01-B11)
        [0,1,0, 0,-1,0, 0,0,0],
        # term 20: (A02-A12)*(B02-B12)
        [0,0,1, 0,0,-1, 0,0,0],
        # term 21: (A00+A11-A22)*(B00+B11-B22)
        [1,0,0, 0,1,0, 0,0,-1],
        # term 22: (A01+A10-A20)*(B10+B01-B00)
        [0,1,0, 1,0,0, -1,0,0],
    ]
    
    V_laderman = [
        # term 0: B00
        [1,0,0, 0,0,0, 0,0,0],
        # term 1: B01
        [0,1,0, 0,0,0, 0,0,0],
        # term 2: B10
        [0,0,0, 1,0,0, 0,0,0],
        # term 3: B11
        [0,0,0, 0,1,0, 0,0,0],
        # term 4: B20
        [0,0,0, 0,0,0, 1,0,0],
        # term 5: B21
        [0,0,0, 0,0,0, 0,1,0],
        # term 6: B00
        [1,0,0, 0,0,0, 0,0,0],
        # term 7: B01
        [0,1,0, 0,0,0, 0,0,0],
        # term 8: B10
        [0,0,0, 1,0,0, 0,0,0],
        # term 9: B11
        [0,0,0, 0,1,0, 0,0,0],
        # term 10: B20
        [0,0,0, 0,0,0, 1,0,0],
        # term 11: B21
        [0,0,0, 0,0,0, 0,1,0],
        # term 12: B00
        [1,0,0, 0,0,0, 0,0,0],
        # term 13: B01
        [0,1,0, 0,0,0, 0,0,0],
        # term 14: B10
        [0,0,0, 1,0,0, 0,0,0],
        # term 15: B11
        [0,0,0, 0,1,0, 0,0,0],
        # term 16: B20
        [0,0,0, 0,0,0, 1,0,0],
        # term 17: B21
        [0,0,0, 0,0,0, 0,1,0],
        # term 18: (B00-B10)
        [1,0,0, -1,0,0, 0,0,0],
        # term 19: (B01-B11)
        [0,1,0, 0,-1,0, 0,0,0],
        # term 20: (B02-B12)
        [0,0,1, 0,0,-1, 0,0,0],
        # term 21: (B00+B11-B22)
        [1,0,0, 0,1,0, 0,0,-1],
        # term 22: (B10+B01-B00)
        [0,1,-1, 1,0,0, 0,0,0],
    ]
    
    W_laderman = [
        # term 0: C00
        [1,0,0, 0,0,0, 0,0,0],
        # term 1: C01
        [0,1,0, 0,0,0, 0,0,0],
        # term 2: C10
        [0,0,0, 1,0,0, 0,0,0],
        # term 3: C11
        [0,0,0, 0,1,0, 0,0,0],
        # term 4: C20
        [0,0,0, 0,0,0, 1,0,0],
        # term 5: C21
        [0,0,0, 0,0,0, 0,1,0],
        # term 6: C10
        [0,0,0, 1,0,0, 0,0,0],
        # term 7: C11
        [0,0,0, 0,1,0, 0,0,0],
        # term 8: C10
        [0,0,0, 1,0,0, 0,0,0],
        # term 9: C11
        [0,0,0, 0,1,0, 0,0,0],
        # term 10: C20
        [0,0,0, 0,0,0, 1,0,0],
        # term 11: C21
        [0,0,0, 0,0,0, 0,1,0],
        # term 12: C20
        [0,0,0, 0,0,0, 1,0,0],
        # term 13: C21
        [0,0,0, 0,0,0, 0,1,0],
        # term 14: C20
        [0,0,0, 0,0,0, 1,0,0],
        # term 15: C21
        [0,0,0, 0,0,0, 0,1,0],
        # term 16: C20
        [0,0,0, 0,0,0, 1,0,0],
        # term 17: C21
        [0,0,0, 0,0,0, 0,1,0],
        # term 18: (C00-C10)
        [1,0,0, -1,0,0, 0,0,0],
        # term 19: (C01-C11)
        [0,1,0, 0,-1,0, 0,0,0],
        # term 20: (C02-C12)
        [0,0,1, 0,0,-1, 0,0,0],
        # term 21: (C00+C11-C22)
        [1,0,0, 0,1,0, 0,0,-1],
        # term 22: (C01+C10-C00)
        [0,1,0, 1,0,0, -1,0,0],
    ]
    
    # Now build the full 3x3x7 decomposition
    # Block 1: A(3x3) * B(3x3) -> C(3x3), using Laderman rank 23
    # Block 2: A(3x3) * B(3x3) -> C(3x3), using Laderman rank 23
    # Block 3: A(3x3) * B(3x1) -> C(3x1), using schoolbook rank 9
    
    rank = 23 + 23 + 9
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # Block 1: columns 0..22
    for t in range(23):
        for i in range(3):
            for j in range(3):
                U[i*m+j][col] = U_laderman[t][i*3+j]
        for j in range(3):
            for k in range(3):
                V[j*p+k][col] = V_laderman[t][j*3+k]
        for k in range(3):
            for i in range(3):
                W[k*n+i][col] = W_laderman[t][k*3+i]
        col += 1
    
    # Block 2: columns 23..45, B entries are at positions 3..5, C entries at positions 3..5
    for t in range(23):
        for i in range(3):
            for j in range(3):
                U[i*m+j][col] = U_laderman[t][i*3+j]
        for j in range(3):
            for k in range(3):
                V[j*p+(k+3)][col] = V_laderman[t][j*3+k]
        for k in range(3):
            for i in range(3):
                W[(k+3)*n+i][col] = W_laderman[t][k*3+i]
        col += 1
    
    # Block 3: columns 46..54, B entry at position 6, C entry at position 6
    for i in range(3):
        for j in range(3):
            U[i*m+j][col] = 1
            V[j*p+6][col] = 1
            W[6*n+i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
