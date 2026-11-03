"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 39 decomposition
    # Split A into A2 (2x2) and A1 (2x1)
    # Split B into B2 (2x7) and B1 (1x7)
    # C = A2 B2 + A1 B1
    
    # A2 B2 part:
    # Split B2 into 3 blocks of 2x2 and 1 block of 2x1
    # Use Strassen (rank 7) for 2x2x2 blocks
    # Use Schoolbook (rank 4) for 2x2x1 block
    # Total rank for A2 B2: 3*7 + 4 = 25
    
    # A1 B1 part:
    # Outer product of 2x1 and 1x7
    # Rank 14
    
    # Total rank 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen decomposition for 2x2x2 block
    # A indices: (0,0), (0,1), (1,0), (1,1) in A2
    # B indices: (0,0), (0,1), (1,0), (1,1) in B2 block
    # C indices: (0,0), (0,1), (1,0), (1,1) in C block
    
    def add_strassen(a_base, b_base, c_base):
        nonlocal col
        # a_base: index offset for A2 rows/cols
        # b_base: index offset for B2 rows/cols
        # c_base: index offset for C rows/cols
        
        # Strassen 7 products
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B00-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B10+B11)
        
        # A2 is 2x2. Indices in full A:
        # A2[0,0] is A[0,0], A2[0,1] is A[0,1]
        # A2[1,0] is A[1,0], A2[1,1] is A[1,1]
        
        # B2 block is 2x2.
        # B2[0,0] is B[0, b_base_col], etc.
        
        # We need to map local indices to global U, V, W.
        
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # Strassen terms:
        # 1. (A00+A11)(B00+B11)
        U[0*3+0][col] = 1
        U[1*3+1][col] = 1
        V[0*7+b_base_col][col] = 1
        V[1*7+b_base_col+1][col] = 1
        # C00 += M1, C01 += M1, C10 += M1, C11 += M1
        # Actually C = [[M1-M4+M6, M5+M7], [M2+M4, M1-M3+M5]]
        # C00: M1 - M4 + M6
        W[b_base_col*2+0][col] = 1
        col += 1
        
        # 2. (A10+A11)B00
        U[1*3+0][col] = 1
        U[1*3+1][col] = 1
        V[0*7+b_base_col][col] = 1
        # C10 += M2, C11 += M2
        W[b_base_col*2+1][col] = 1
        col += 1
        
        # 3. A00(B10-B11)
        U[0*3+0][col] = 1
        V[1*7+b_base_col][col] = 1
        V[1*7+b_base_col+1][col] = -1
        # C11 -= M3
        W[b_base_col*2+1][col] = -1
        col += 1
        
        # 4. A11(B00-B10)
        U[1*3+1][col] = 1
        V[0*7+b_base_col][col] = 1
        V[1*7+b_base_col][col] = -1
        # C00 -= M4, C10 += M4
        W[b_base_col*2+0][col] = -1
        W[b_base_col*2+1][col] = 1
        col += 1
        
        # 5. (A00+A01)B11
        U[0*3+0][col] = 1
        U[0*3+1][col] = 1
        V[1*7+b_base_col+1][col] = 1
        # C01 += M5, C11 += M5
        W[b_base_col*2+0][col] = 1
        W[b_base_col*2+1][col] = 1
        col += 1
        
        # 6. (A10-A00)(B00+B10)
        U[1*3+0][col] = 1
        U[0*3+0][col] = -1
        V[0*7+b_base_col][col] = 1
        V[1*7+b_base_col][col] = 1
        # C00 += M6
        W[b_base_col*2+0][col] = 1
        col += 1
        
        # 7. (A01-A11)(B10+B11)
        U[0*3+1][col] = 1
        U[1*3+1][col] = -1
        V[1*7+b_base_col][col] = 1
        V[1*7+b_base_col+1][col] = 1
        # C01 += M7
        W[b_base_col*2+0][col] = 1
        col += 1

    # Add Strassen for 3 blocks of B2
    for k in range(3):
        add_strassen(0, k*2, k*2)
        
    # Add Schoolbook for last block of B2 (2x1)
    # A2 (2x2) * B2_last (2x1)
    # C2_last (2x1)
    # c0 = A00*B00 + A01*B10
    # c1 = A10*B00 + A11*B10
    
    # Term 1: A00 B00
    U[0*3+0][col] = 1
    V[0*7+6][col] = 1
    W[6*2+0][col] = 1
    col += 1
    
    # Term 2: A01 B10
    U[0*3+1][col] = 1
    V[1*7+6][col] = 1
    W[6*2+0][col] = 1
    col += 1
    
    # Term 3: A10 B00
    U[1*3+0][col] = 1
    V[0*7+6][col] = 1
    W[6*2+1][col] = 1
    col += 1
    
    # Term 4: A11 B10
    U[1*3+1][col] = 1
    V[1*7+6][col] = 1
    W[6*2+1][col] = 1
    col += 1
    
    # Now A1 B1 part
    # A1 is col 2 of A. B1 is row 2 of B.
    # C += A1 B1
    # C[i, k] += A[i, 2] B[2, k]
    for i in range(2):
        for k in range(7):
            U[i*3+2][col] = 1
            V[2*7+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
