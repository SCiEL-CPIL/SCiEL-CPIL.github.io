"""Strassen-based rank-28 decomposition for 2x4 * 4x4 matrix multiplication."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-28 exact tensor decomposition of the 2x4 * 4x4 matrix product.
    
    Strategy: Partition the 2x4 matrix A into two 2x2 blocks:
      A = [A0 | A1] where A0 = A[:, 0:2], A1 = A[:, 2:4]
    Partition the 4x4 matrix B into four 2x2 blocks:
      B = [[B00, B01], [B10, B11]] where
        B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
        B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    
    The product C = A * B is a 2x4 matrix:
      C[:, 0:2] = A0 * B00 + A1 * B10
      C[:, 2:4] = A0 * B01 + A1 * B11
    
    So we need 4 independent 2x2 * 2x2 matrix products.
    Each 2x2 * 2x2 product uses the Strassen algorithm with rank 7.
    Total rank = 4 * 7 = 28.
    
    Strassen's algorithm for computing P = X * Y (both 2x2):
      M1 = (X00 + X11) * (Y00 + Y11)
      M2 = (X10 + X11) * Y00
      M3 = X00 * (Y10 - Y00)
      M4 = X11 * (Y11 - Y01)
      M5 = (X00 + X01) * Y11
      M6 = (X10 - X00) * (Y00 + Y01)
      M7 = (X01 - X11) * (Y10 + Y11)
      
      P00 = M1 + M4 - M5 + M7
      P01 = M3 + M5
      P10 = M2 + M4
      P11 = M1 - M2 + M3 + M6
    
    We define the factor matrices U, V, W with the convention:
      U has shape (n*m, rank) = (8, 28)  -- indexed by [i*m + j, col]
      V has shape (m*p, rank) = (16, 28) -- indexed by [j*p + k, col]
      W has shape (p*n, rank) = (8, 28)  -- indexed by [k*n + i, col]
    """
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28
    
    # Strassen factors for a single 2x2*2x2 product.
    # For each of the 7 Strassen terms, we have:
    #   U contribution: entries in A (2x2) with coefficients
    #   V contribution: entries in B (2x2) with coefficients
    #   W contribution: entries in C (2x2) with coefficients
    #
    # The 2x2 matrices are indexed as:
    #   A: A[0,0], A[0,1], A[1,0], A[1,1]  -> positions 0,1,2,3 in U's 2x2 block
    #   B: B[0,0], B[0,1], B[1,0], B[1,1]  -> positions 0,1,2,3 in V's 2x2 block
    #   C: C[0,0], C[0,1], C[1,0], C[1,1]  -> positions 0,1,2,3 in W's 2x2 block
    #
    # Strassen terms (a_factors, b_factors, c_factors) where each is list of (index, coeff):
    strassen_terms = [
        # M1 = (X00+X11)*(Y00+Y11) -> contributes to P00(+1), P11(+1)
        ([((0,0),1), ((1,1),1)],
         [((0,0),1), ((1,1),1)],
         [((0,0),1), ((1,1),1)]),
        # M2 = (X10+X11)*Y00 -> contributes to P10(+1), P11(-1)
        ([((1,0),1), ((1,1),1)],
         [((0,0),1)],
         [((1,0),1), ((1,1),-1)]),
        # M3 = X00*(Y10-Y00) -> contributes to P01(+1), P11(+1)
        ([((0,0),1)],
         [((1,0),1), ((0,0),-1)],
         [((0,1),1), ((1,1),1)]),
        # M4 = X11*(Y11-Y01) -> contributes to P00(+1), P10(+1)
        ([((1,1),1)],
         [((1,1),1), ((0,1),-1)],
         [((0,0),1), ((1,0),1)]),
        # M5 = (X00+X01)*Y11 -> contributes to P00(-1), P01(+1)
        ([((0,0),1), ((0,1),1)],
         [((1,1),1)],
         [((0,0),-1), ((0,1),1)]),
        # M6 = (X10-X00)*(Y00+Y01) -> contributes to P11(+1)
        ([((1,0),1), ((0,0),-1)],
         [((0,0),1), ((0,1),1)],
         [((1,1),1)]),
        # M7 = (X01-X11)*(Y10+Y11) -> contributes to P00(+1)
        ([((0,1),1), ((1,1),-1)],
         [((1,0),1), ((1,1),1)],
         [((0,0),1)]),
    ]
    
    # The 4 block products needed:
    # C[:, 0:2] = A0*B00 + A1*B10
    # C[:, 2:4] = A0*B01 + A1*B11
    #
    # Each block product is a 2x2 * 2x2 multiplication.
    # We define them as (a_block, b_block, c_block) where:
    #   a_block: which 2x2 block of A (col offset in A)
    #   b_block: which 2x2 block of B (row offset, col offset in B)
    #   c_block: which 2x2 block of C (col offset in C)
    #
    # A is 2x4: A[:, 0:2] is block 0, A[:, 2:4] is block 1
    # B is 4x4: B[0:2,0:2]=B00, B[0:2,2:4]=B01, B[2:4,0:2]=B10, B[2:4,2:4]=B11
    # C is 2x4: C[:, 0:2] is block 0, C[:, 2:4] is block 1
    #
    # Product 1: A0 * B00 -> C[:, 0:2]
    # Product 2: A1 * B10 -> C[:, 0:2]
    # Product 3: A0 * B01 -> C[:, 2:4]
    # Product 4: A1 * B11 -> C[:, 2:4]
    
    # (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    block_products = [
        (0, 0, 0, 0),  # A0 * B00 -> C[:, 0:2]
        (2, 2, 0, 0),  # A1 * B10 -> C[:, 0:2]
        (0, 0, 2, 2),  # A0 * B01 -> C[:, 2:4]
        (2, 2, 2, 2),  # A1 * B11 -> C[:, 2:4]
    ]
    
    for prod_idx, (a_off, b_row_off, b_col_off, c_off) in enumerate(block_products):
        base_col = prod_idx * 7
        for s_idx in range(7):
            col = base_col + s_idx
            a_factors, b_factors, c_factors = strassen_terms[s_idx]
            
            # Fill U: U[i*m + j, col] where i in {0,1}, j in {0,1,2,3}
            # For this 2x2 block of A, the columns are a_off and a_off+1
            for (a_row, a_col_local), coeff in a_factors:
                i = a_row
                j = a_off + a_col_local
                U[i * m + j][col] += coeff
            
            # Fill V: V[j*p + k, col] where j in {0,1,2,3}, k in {0,1,2,3}
            # For this 2x2 block of B, rows are b_row_off, b_row_off+1; cols are b_col_off, b_col_off+1
            for (b_row, b_col_local), coeff in b_factors:
                j = b_row_off + b_row
                k = b_col_off + b_col_local
                V[j * p + k][col] += coeff
            
            # Fill W: W[k*n + i, col] where k in {0,1,2,3}, i in {0,1}
            # For this 2x2 block of C, cols are c_off, c_off+1; rows are 0, 1
            for (c_row, c_col_local), coeff in c_factors:
                i = c_row
                k = c_off + c_col_local
                W[k * n + i][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
