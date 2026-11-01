"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 38
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Function to add a 2x2x2 Strassen decomposition
    # a_indices: list of 4 U indices for a00, a01, a10, a11
    # b_indices: list of 4 V indices for b00, b01, b10, b11
    # c_indices: list of 4 W indices for c00, c01, c10, c11
    def add_strassen(a_idx, b_idx, c_idx):
        nonlocal col
        # M1 = (a00+a11)(b00+b11)
        U[a_idx[0]][col] = 1
        U[a_idx[3]][col] = 1
        V[b_idx[0]][col] = 1
        V[b_idx[3]][col] = 1
        W[c_idx[0]][col] = 1
        W[c_idx[3]][col] = 1
        W[c_idx[1]][col] = -1
        W[c_idx[2]][col] = -1
        col += 1
        
        # M2 = (a10+a11)b00
        U[a_idx[2]][col] = 1
        U[a_idx[3]][col] = 1
        V[b_idx[0]][col] = 1
        W[c_idx[2]][col] = 1
        W[c_idx[3]][col] = -1
        col += 1
        
        # M3 = a00(b01-b11)
        U[a_idx[0]][col] = 1
        V[b_idx[1]][col] = 1
        V[b_idx[3]][col] = -1
        W[c_idx[1]][col] = 1
        W[c_idx[3]][col] = 1
        col += 1
        
        # M4 = a11(b00-b10)
        U[a_idx[3]][col] = 1
        V[b_idx[0]][col] = 1
        V[b_idx[2]][col] = -1
        W[c_idx[0]][col] = 1
        W[c_idx[2]][col] = 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[a_idx[0]][col] = 1
        U[a_idx[1]][col] = 1
        V[b_idx[3]][col] = 1
        W[c_idx[0]][col] = -1
        W[c_idx[1]][col] = 1
        col += 1
        
        # M6 = (a10-a11)(b00+b01)
        U[a_idx[2]][col] = 1
        U[a_idx[3]][col] = -1
        V[b_idx[0]][col] = 1
        V[b_idx[1]][col] = 1
        W[c_idx[3]][col] = -1
        col += 1
        
        # M7 = (a01-a10)(b10+b11)
        U[a_idx[1]][col] = 1
        U[a_idx[2]][col] = -1
        V[b_idx[2]][col] = 1
        V[b_idx[3]][col] = 1
        W[c_idx[0]][col] = 1
        col += 1

    # Function to add a 2x2x3 Schoolbook decomposition
    # a_indices: list of 4 U indices
    # b_indices: list of 6 V indices for b00..b02, b10..b12
    # c_indices: list of 6 W indices for c00..c02, c10..c12
    def add_schoolbook_2x2x3(a_idx, b_idx, c_idx):
        nonlocal col
        # 12 multiplications
        # a00*b00, a00*b01, a00*b02
        # a01*b10, a01*b11, a01*b12
        # a10*b00, a10*b01, a10*b02
        # a11*b10, a11*b11, a11*b12
        
        # C00 = a00*b00 + a01*b10
        # C01 = a00*b01 + a01*b11
        # C02 = a00*b02 + a01*b12
        # C10 = a10*b00 + a11*b10
        # C11 = a10*b01 + a11*b11
        # C12 = a10*b02 + a11*b12
        
        for k in range(3):
            # a00 * b0k
            U[a_idx[0]][col] = 1
            V[b_idx[k]][col] = 1
            W[c_idx[k]][col] = 1
            col += 1
            # a01 * b1k
            U[a_idx[1]][col] = 1
            V[b_idx[3+k]][col] = 1
            W[c_idx[k]][col] = 1
            col += 1
            # a10 * b0k
            U[a_idx[2]][col] = 1
            V[b_idx[k]][col] = 1
            W[c_idx[3+k]][col] = 1
            col += 1
            # a11 * b1k
            U[a_idx[3]][col] = 1
            V[b_idx[3+k]][col] = 1
            W[c_idx[3+k]][col] = 1
            col += 1

    # Block 1: A1 (cols 0,1), B1 (rows 0,1)
    # A1 U indices: 0,1,4,5
    # B1 V indices: 0,1,2,3,4 (row 0), 5,6,7,8,9 (row 1)
    # C1 W indices: 0,1 (row 0), 2,3 (row 1)
    
    # Strassen on cols 0,1 of B1
    a1_idx = [0, 1, 4, 5]
    b1_strassen_idx = [0, 1, 5, 6] # b00, b01, b10, b11
    c1_strassen_idx = [0, 1, 2, 3] # c00, c01, c10, c11
    add_strassen(a1_idx, b1_strassen_idx, c1_strassen_idx)
    
    # Schoolbook on cols 2,3,4 of B1
    b1_school_idx = [2, 3, 4, 7, 8, 9] # b02, b03, b04, b12, b13, b14
    c1_school_idx = [4, 5, 6, 7, 8, 9] # c02, c03, c04, c12, c13, c14
    add_schoolbook_2x2x3(a1_idx, b1_school_idx, c1_school_idx)
    
    # Block 2: A2 (cols 2,3), B2 (rows 2,3)
    # A2 U indices: 2,3,6,7
    # B2 V indices: 10,11,12,13,14 (row 2), 15,16,17,18,19 (row 3)
    # C2 W indices: 4,5 (row 2? No, C is 2x5. Rows of C correspond to rows of A? No.
    # C[k*n+i]. k is row of B (0..3), i is row of A (0..1).
    # B2 rows are 2,3. So k=2,3.
    # C2 W indices: k=2,3. i=0,1.
    # W indices: 2*2+0=4, 2*2+1=5, 3*2+0=6, 3*2+1=7.
    
    a2_idx = [2, 3, 6, 7]
    b2_strassen_idx = [10, 11, 15, 16] # b20, b21, b30, b31
    c2_strassen_idx = [4, 5, 6, 7] # c20, c21, c30, c31
    add_strassen(a2_idx, b2_strassen_idx, c2_strassen_idx)
    
    b2_school_idx = [12, 13, 14, 17, 18, 19] # b22, b23, b24, b32, b33, b34
    c2_school_idx = [8, 9, 10, 11, 12, 13] # c22, c23, c24, c32, c33, c34
    add_schoolbook_2x2x3(a2_idx, b2_school_idx, c2_school_idx)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
