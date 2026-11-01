import time

def construct():
    """
    Rank-42 exact decomposition for 2x4 @ 4x6 matrix multiplication.
    
    Strategy: Split the 4x4 middle dimension into two 2x2 blocks (left and right halves).
    This gives C = A_left @ B_top + A_right @ B_bottom, where each term is a 2x2 @ 2x6 multiplication.
    Each 2x2 @ 2x6 is decomposed into three 2x2 @ 2x2 blocks using Strassen's algorithm (rank 7 each).
    Total rank per 2x2 @ 2x6 block = 3 * 7 = 21.
    Total rank = 21 + 21 = 42.
    
    U: (n*m, rank) = (8, 42)
    V: (m*p, rank) = (24, 42)
    W: (p*n, rank) = (12, 42)
    
    W index convention: W[k*n+i, r] where k is column index (0..5), i is row index (0..1)
    So W index = k*2 + i for output C[i][k]
    """
    n, m, p = 2, 4, 6
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Left block: A_left is rows 0-1, cols 0-1 of A (2x2)
    # B_top is rows 0-1, cols 0-5 of B (2x6)
    # Output: C_left[i][j] for i in 0..1, j in 0..5
    # C_left[i][j] = A[0][0]*B[0][j] + A[0][1]*B[1][j] for i=0
    #                A[1][0]*B[0][j] + A[1][1]*B[1][j] for i=1
    
    # Right block: A_right is rows 0-1, cols 2-3 of A (2x2)
    # B_bottom is rows 2-3, cols 0-5 of B (2x6)
    # Output: C_right[i][j] for i in 0..1, j in 0..5
    
    col = 0
    
    # Strassen 2x2x2 core: 7 multiplications
    # For X @ Y where X,Y are 2x2:
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y01-Y11)
    # M4 = X11*(Y10-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    # Result:
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    
    # We decompose 2x4 @ 4x6 as:
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    # Each term is 2x2 @ 2x6
    # Split 2x6 into three 2x2 blocks: cols 0-1, 2-3, 4-5
    # So 2x2 @ 2x6 = concat of three 2x2 @ 2x2 multiplications
    # Each 2x2 @ 2x2 uses Strassen rank 7
    # Total: 2 blocks * 3 sub-blocks * 7 = 42
    
    # Define Strassen pattern for 2x2 @ 2x2
    # U indices: row*2+col for A matrix (2x2)
    # V indices: row*2+col for B matrix (2x2)
    # W indices: col*2+row for output (following convention W[k*n+i,r] where k=col, i=row)
    
    # Strassen 7 terms with (u_coeff, v_coeff, w_coeff) for each output element
    # For each of the 7 multiplications, we need:
    # - which elements of A are combined (with signs)
    # - which elements of B are combined (with signs)
    # - how the product contributes to each output element
    
    # M1: (A00+A11)*(B00+B11) -> contributes to Z00(+1), Z11(+1)
    # M2: (A10+A11)*B00 -> contributes to Z10(+1), Z11(-1)
    # M3: A00*(B01-B11) -> contributes to Z01(+1), Z11(+1)
    # M4: A11*(B10-B00) -> contributes to Z00(+1), Z10(+1)
    # M5: (A00+A01)*B11 -> contributes to Z00(-1), Z01(+1)
    # M6: (A10-A00)*(B00+B01) -> contributes to Z11(+1)
    # M7: (A00+A11)*(B10+B11) -> wait, M7 = (A01-A11)*(B10+B11) -> contributes to Z00(+1)
    
    # Let me define each Strassen term as:
    # (list of (a_row, a_col, sign) for A, list of (b_row, b_col, sign) for B, list of (out_row, out_col, sign) for output)
    
    strassen_terms = [
        # M1: (A00+A11)*(B00+B11)
        {
            'A': [(0,0,1), (1,1,1)],
            'B': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]  # Z00 += M1, Z11 += M1
        },
        # M2: (A10+A11)*B00
        {
            'A': [(1,0,1), (1,1,1)],
            'B': [(0,0,1)],
            'W': [(1,0,1), (1,1,-1)]  # Z10 += M2, Z11 -= M2
        },
        # M3: A00*(B01-B11)
        {
            'A': [(0,0,1)],
            'B': [(0,1,1), (1,1,-1)],
            'W': [(0,1,1), (1,1,1)]  # Z01 += M3, Z11 += M3
        },
        # M4: A11*(B10-B00)
        {
            'A': [(1,1,1)],
            'B': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (1,0,1)]  # Z00 += M4, Z10 += M4
        },
        # M5: (A00+A01)*B11
        {
            'A': [(0,0,1), (0,1,1)],
            'B': [(1,1,1)],
            'W': [(0,0,-1), (0,1,1)]  # Z00 -= M5, Z01 += M5
        },
        # M6: (A10-A00)*(B00+B01)
        {
            'A': [(1,0,1), (0,0,-1)],
            'B': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]  # Z11 += M6
        },
        # M7: (A01-A11)*(B10+B11)
        {
            'A': [(0,1,1), (1,1,-1)],
            'B': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]  # Z00 += M7
        }
    ]
    
    # Helper: apply Strassen for a 2x2 @ 2x2 sub-problem
    # a_base: (row_offset, col_offset) in A (2x4 matrix, indexed as [i][j])
    # b_base: (row_offset, col_offset) in B (4x6 matrix, indexed as [i][j])
    # out_base: (row_offset, col_offset) in output C (2x6, indexed as [i][j])
    def apply_strassen(a_base, b_base, out_base, col_start):
        nonlocal col
        for term in strassen_terms:
            # Set U coefficients
            for (ar, ac, sign) in term['A']:
                # A element at (a_base[0]+ar, a_base[1]+ac) in the 2x4 matrix
                # U index: row * m + col = (a_base[0]+ar) * 4 + (a_base[1]+ac)
                u_idx = (a_base[0] + ar) * m + (a_base[1] + ac)
                U[u_idx][col] += sign
            
            # Set V coefficients
            for (br, bc, sign) in term['B']:
                # B element at (b_base[0]+br, b_base[1]+bc) in the 4x6 matrix
                # V index: row * p + col = (b_base[0]+br) * 6 + (b_base[1]+bc)
                v_idx = (b_base[0] + br) * p + (b_base[1] + bc)
                V[v_idx][col] += sign
            
            # Set W coefficients
            for (or_, oc, sign) in term['W']:
                # Output element at (out_base[0]+or_, out_base[1]+oc) in C (2x6)
                # W index: col * n + row = (out_base[1]+oc) * 2 + (out_base[0]+or_)
                w_idx = (out_base[1] + oc) * n + (out_base[0] + or_)
                W[w_idx][col] += sign
            
            col += 1
    
    # Left half: A[:,0:2] @ B[0:2,:]
    # Split into 3 sub-blocks of B: cols 0-1, 2-3, 4-5
    for sub in range(3):
        # B sub-block: rows 0-1, cols 2*sub to 2*sub+1
        b_col_offset = 2 * sub
        # Output: cols 2*sub to 2*sub+1
        out_col_offset = 2 * sub
        apply_strassen((0, 0), (0, b_col_offset), (0, out_col_offset), col)
    
    # Right half: A[:,2:4] @ B[2:4,:]
    # Split into 3 sub-blocks of B: cols 0-1, 2-3, 4-5
    for sub in range(3):
        # B sub-block: rows 2-3, cols 2*sub to 2*sub+1
        b_col_offset = 2 * sub
        # Output: cols 2*sub to 2*sub+1
        out_col_offset = 2 * sub
        apply_strassen((0, 2), (2, b_col_offset), (0, out_col_offset), col)
    
    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
