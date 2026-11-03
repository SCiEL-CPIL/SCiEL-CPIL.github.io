"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-34 decomposition of 3x3 * 3x4 matrix multiplication using Strassen
    on the 2x2x2 sub-blocks.
    
    A (3x3) split as: A00(2x2), A01(2x1), A10(1x2), A11(1x1)
    B (3x4) split as: B00(2x2), B01(2x2), B10(1x2), B11(1x2)
    
    C00(2x2) = A00*B00 (Strassen: 7) + A01*B10 (schoolbook: 4)
    C01(2x2) = A00*B01 (Strassen: 7) + A01*B11 (schoolbook: 4)
    C10(1x2) = A10*B00 (schoolbook: 4) + A11*B10 (schoolbook: 2)
    C11(1x2) = A10*B01 (schoolbook: 4) + A11*B11 (schoolbook: 2)
    Total rank = 7+4+7+4+4+2+4+2 = 34
    """
    n, m, p = 3, 3, 4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]  # 9 x 34
    V = [[0] * rank for _ in range(m * p)]  # 12 x 34
    W = [[0] * rank for _ in range(p * n)]  # 12 x 34
    
    col = 0
    
    # Indexing:
    # U[i*3+j] for A(i,j)
    # V[j*4+k] for B(j,k)
    # W[k*3+i] for C(i,k)  (W[k*n+i, r])
    
    def add_term(u_indices, v_indices, w_indices, coeff=1):
        nonlocal col
        for u_idx in u_indices:
            U[u_idx][col] += coeff
        for v_idx in v_indices:
            V[v_idx][col] += coeff
        for w_idx in w_indices:
            W[w_idx][col] += coeff
        col += 1
    
    # Strassen helper for 2x2*2x2->2x2
    def add_strassen(a_tl, b_tl, c_tl):
        """a_tl, b_tl, c_tl are (row, col) top-left corners in A, B, C"""
        nonlocal col
        # A elements: (a_tl_r, a_tl_c), (a_tl_r, a_tl_c+1), (a_tl_r+1, a_tl_c), (a_tl_r+1, a_tl_c+1)
        # B elements: (b_tl_r, b_tl_c), (b_tl_r, b_tl_c+1), (b_tl_r+1, b_tl_c), (b_tl_r+1, b_tl_c+1)
        # C elements: (c_tl_r, c_tl_c), (c_tl_r, c_tl_c+1), (c_tl_r+1, c_tl_c), (c_tl_r+1, c_tl_c+1)
        
        a00 = a_tl[0]*3 + a_tl[1]
        a01 = a_tl[0]*3 + (a_tl[1]+1)
        a10 = (a_tl[0]+1)*3 + a_tl[1]
        a11 = (a_tl[0]+1)*3 + (a_tl[1]+1)
        
        b00 = b_tl[0]*4 + b_tl[1]
        b01 = b_tl[0]*4 + (b_tl[1]+1)
        b10 = (b_tl[0]+1)*4 + b_tl[1]
        b11 = (b_tl[0]+1)*4 + (b_tl[1]+1)
        
        c00 = c_tl[1]*3 + c_tl[0]
        c01 = (c_tl[1]+1)*3 + c_tl[0]
        c10 = c_tl[1]*3 + (c_tl[0]+1)
        c11 = (c_tl[1]+1)*3 + (c_tl[0]+1)
        
        # M1 = (A00+A11)*(B00+B11) -> C00 += 1, C11 += 1
        add_term([a00, a11], [b00, b11], [c00, c11], 1)
        # M2 = (A10+A11)*B00 -> C10 += 1
        add_term([a10, a11], [b00], [c10], 1)
        # M3 = A00*(B01-B11) -> C01 += 1
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1
        col += 1
        # M4 = A11*(B10-B00) -> C00 += 1, C10 += 1
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        # M5 = (A00+A10)*B11 -> C00 -= 1, C01 += 1
        U[a00][col] += 1; U[a10][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        # M6 = (A01-A11)*(B10+B11) -> C11 += 1
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c11][col] += 1
        col += 1
        # M7 = (A10-A00)*(B00+A01) -> C00 += 1
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c00][col] += 1
        col += 1
    
    # C00 = A00*B00 (Strassen) + A01*B10 (schoolbook)
    add_strassen((0,0), (0,0), (0,0))  # 7 terms
    
    # A01*B10: A(0,2)*B(2,0)->C(0,0), A(0,2)*B(2,1)->C(0,1), A(1,2)*B(2,0)->C(1,0), A(1,2)*B(2,1)->C(1,1)
    add_term([2], [8], [0], 1)
    add_term([2], [9], [3], 1)
    add_term([5], [8], [1], 1)
    add_term([5], [9], [4], 1)
    
    # C01 = A00*B01 (Strassen) + A01*B11 (schoolbook)
    add_strassen((0,0), (0,2), (0,2))  # 7 terms
    
    # A01*B11: A(0,2)*B(2,2)->C(0,2), A(0,2)*B(2,3)->C(0,3), A(1,2)*B(2,2)->C(1,2), A(1,2)*B(2,3)->C(1,3)
    add_term([2], [10], [6], 1)
    add_term([2], [11], [7], 1)
    add_term([5], [10], [9], 1)
    add_term([5], [11], [10], 1)
    
    # C10 = A10*B00 (schoolbook) + A11*B10 (schoolbook)
    # A10*B00: A(2,0)*B(0,0)->C(2,0), A(2,0)*B(0,1)->C(2,1), A(2,1)*B(1,0)->C(2,0), A(2,1)*B(1,1)->C(2,1)
    add_term([6], [0], [2], 1)
    add_term([6], [1], [5], 1)
    add_term([7], [4], [2], 1)
    add_term([7], [5], [5], 1)
    
    # A11*B10: A(2,2)*B(2,0)->C(2,0), A(2,2)*B(2,1)->C(2,1)
    add_term([8], [8], [2], 1)
    add_term([8], [9], [5], 1)
    
    # C11 = A10*B01 (schoolbook) + A11*B11 (schoolbook)
    # A10*B01: A(2,0)*B(0,2)->C(2,2), A(2,0)*B(0,3)->C(2,3), A(2,1)*B(1,2)->C(2,2), A(2,1)*B(1,3)->C(2,3)
    add_term([6], [2], [8], 1)
    add_term([6], [3], [11], 1)
    add_term([7], [6], [8], 1)
    add_term([7], [7], [11], 1)
    
    # A11*B11: A(2,2)*B(2,2)->C(2,2), A(2,2)*B(2,3)->C(2,3)
    add_term([8], [10], [8], 1)
    add_term([8], [11], [11], 1)
    
    assert col == 34, f"Expected 34 terms, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

