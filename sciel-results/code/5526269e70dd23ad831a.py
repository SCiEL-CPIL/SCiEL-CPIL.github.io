"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    # We will construct a rank-44 decomposition.
    # Split n (3) into 2 and 1.
    # Part 1: 1 x 4 x 4 (schoolbook, rank 16)
    # Part 2: 2 x 4 x 4 (Strassen on 2x2x2 blocks, rank 28)
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 3*4 = 12 rows
    V = [[0] * rank for _ in range(m * p)]  # 4*4 = 16 rows
    W = [[0] * rank for _ in range(p * n)]  # 4*3 = 12 rows
    
    col = 0
    
    # Part 1: 1 x 4 x 4 schoolbook
    # This corresponds to the 3rd row of A (index 2) and all cols/rows.
    # A_row_2 (1x4) * B (4x4) -> C_row_2 (1x4)
    for j in range(4):
        for k in range(4):
            # U index: row 2 of A, col j -> 2*4 + j
            U[2 * 4 + j][col] = 1
            # V index: row j of B, col k -> j*4 + k
            V[j * 4 + k][col] = 1
            # W index: row k of C, row 2 of C -> k*3 + 2
            W[k * 3 + 2][col] = 1
            col += 1
    # col is now 16
    
    # Part 2: 2 x 4 x 4
    # Rows 0, 1 of A.
    # Split inner dim (4) into 2+2.
    # Split output dim (4) into 2+2.
    # This creates 4 blocks of 2x2x2.
    # Block (i_inner, k_out) where i_inner in [0,1], k_out in [0,1]
    # i_inner 0 -> inner cols 0,1
    # i_inner 1 -> inner cols 2,3
    # k_out 0 -> output cols 0,1
    # k_out 1 -> output cols 2,3
    
    for i_inner in range(2):
        for k_out in range(2):
            # Strassen for 2x2x2
            # A_sub is 2x2 (rows 0,1 of A, cols 2*i_inner, 2*i_inner+1)
            # B_sub is 2x2 (rows 2*i_inner, 2*i_inner+1 of B, cols 2*k_out, 2*k_out+1)
            # C_sub is 2x2 (rows 0,1 of C, cols 2*k_out, 2*k_out+1)
            
            # Strassen terms:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # We need to add these to U, V, W.
            
            # Helper to add a term
            def add_term(u_coeff, v_coeff, w_coeff, row_u, col_u, row_v, col_v, row_w, col_w):
                nonlocal col
                U[row_u][col] += u_coeff
                V[row_v][col] += v_coeff
                W[row_w][col] += w_coeff
                col += 1
            
            # Indices for A_sub (rows 0,1; cols 2*i_inner, 2*i_inner+1)
            # A00: row 0, col 2*i_inner
            # A01: row 0, col 2*i_inner+1
            # A10: row 1, col 2*i_inner
            # A11: row 1, col 2*i_inner+1
            
            # Indices for B_sub (rows 2*i_inner, 2*i_inner+1; cols 2*k_out, 2*k_out+1)
            # B00: row 2*i_inner, col 2*k_out
            # B01: row 2*i_inner, col 2*k_out+1
            # B10: row 2*i_inner+1, col 2*k_out
            # B11: row 2*i_inner+1, col 2*k_out+1
            
            # Indices for C_sub (rows 0,1; cols 2*k_out, 2*k_out+1)
            # C00: row 0, col 2*k_out
            # C01: row 0, col 2*k_out+1
            # C10: row 1, col 2*k_out
            # C11: row 1, col 2*k_out+1
            
            # Term 1: (A00+A11)(B00+B11)
            # U: A00=1, A11=1
            # V: B00=1, B11=1
            # W: C00=1, C11=1
            # We need to distribute this product.
            # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
            # This is 4 schoolbook terms if expanded, but Strassen treats it as 1 multiplication.
            # In the tensor decomposition, each "product" M_i is a rank-1 term?
            # No, M_i is a product of linear forms.
            # The rank-1 term corresponds to $u \otimes v \otimes w$.
            # For Strassen, the 7 terms are the rank-1 components.
            # So for M1, we have one rank-1 term where:
            # u = A00 + A11 (coeff 1 for A00, 1 for A11)
            # v = B00 + B11 (coeff 1 for B00, 1 for B11)
            # w = C00 + C11 (coeff 1 for C00, 1 for C11)
            
            # Term 1
            U[0 * 4 + 2 * i_inner][col] += 1
            U[1 * 4 + 2 * i_inner + 1][col] += 1
            V[2 * i_inner * 4 + 2 * k_out][col] += 1
            V[(2 * i_inner + 1) * 4 + 2 * k_out + 1][col] += 1
            W[0 * 3 + 2 * k_out][col] += 1
            W[1 * 3 + 2 * k_out + 1][col] += 1
            col += 1
            
            # Term 2: (A10+A11)B00
            U[1 * 4 + 2 * i_inner][col] += 1
            U[1 * 4 + 2 * i_inner + 1][col] += 1
            V[2 * i_inner * 4 + 2 * k_out][col] += 1
            W[1 * 3 + 2 * k_out][col] += 1
            col += 1
            
            # Term 3: A00(B01-B11)
            U[0 * 4 + 2 * i_inner][col] += 1
            V[2 * i_inner * 4 + 2 * k_out + 1][col] += 1
            V[(2 * i_inner + 1) * 4 + 2 * k_out + 1][col] += -1
            W[0 * 3 + 2 * k_out + 1][col] += 1
            col += 1
            
            # Term 4: A11(B10-B00)
            U[1 * 4 + 2 * i_inner + 1][col] += 1
            V[(2 * i_inner + 1) * 4 + 2 * k_out][col] += 1
            V[2 * i_inner * 4 + 2 * k_out][col] += -1
            W[1 * 3 + 2 * k_out][col] += 1
            col += 1
            
            # Term 5: (A00+A01)B11
            U[0 * 4 + 2 * i_inner][col] += 1
            U[0 * 4 + 2 * i_inner + 1][col] += 1
            V[(2 * i_inner + 1) * 4 + 2 * k_out + 1][col] += 1
            W[1 * 3 + 2 * k_out + 1][col] += 1
            col += 1
            
            # Term 6: (A10-A00)(B00+B01)
            U[1 * 4 + 2 * i_inner][col] += 1
            U[0 * 4 + 2 * i_inner][col] += -1
            V[2 * i_inner * 4 + 2 * k_out][col] += 1
            V[2 * i_inner * 4 + 2 * k_out + 1][col] += 1
            W[1 * 3 + 2 * k_out + 1][col] += 1
            col += 1
            
            # Term 7: (A01-A11)(B10+B11)
            U[0 * 4 + 2 * i_inner + 1][col] += 1
            U[1 * 4 + 2 * i_inner + 1][col] += -1
            V[(2 * i_inner + 1) * 4 + 2 * k_out][col] += 1
            V[(2 * i_inner + 1) * 4 + 2 * k_out + 1][col] += 1
            W[1 * 3 + 2 * k_out + 1][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
