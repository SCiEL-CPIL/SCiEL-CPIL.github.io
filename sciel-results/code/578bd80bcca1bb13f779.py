"""
Constructs a rank-98 tensor decomposition for 4x4 x 4x7 matrix multiplication
using recursive Strassen's algorithm (rank 7 for 2x2x2) applied to 4x4x4 blocks.

The approach:
1. Decompose 4x4x4 multiplication using Strassen's algorithm recursively:
   - Split 4x4 matrices into 2x2 blocks of 2x2 submatrices
   - Each 2x2 block multiplication uses Strassen's rank-7 decomposition
   - Total rank for 4x4x4: 7 * 7 = 49
2. For 4x4 x 4x7, split the 4x7 matrix into two 4x4 parts (last column zero in second part)
   - First 4x4 part uses ranks 0..48
   - Second 4x4 part (with last column zero) uses ranks 49..97
   - Total rank: 98

This is significantly better than schoolbook rank of 112.
"""

DIMENSIONS = [4, 4, 7]

# Strassen's rank-7 decomposition for 2x2x2 matrix multiplication
# Indices: U[i*m+j], V[j*p+k], W[k*n+i] for n=m=p=2
# Strassen's 7 products:
# P1 = (A00+A11)*(B00+B11)
# P2 = (A10+A11)*B00
# P3 = A00*(B01-B11)
# P4 = A11*(B10-B00)
# P5 = (A00+A01)*B11
# P6 = (A10-A00)*(B00+B01)
# P7 = (A01-A11)*(B10+B11)
#
# Then:
# C00 = P1 + P4 - P5 + P7
# C01 = P3 + P5
# C10 = P2 + P4
# C11 = P1 - P2 + P3 + P6

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2 x 2x2 -> 2x2 matrix multiplication.
    U shape: (4, 7) for input A (2x2)
    V shape: (4, 7) for input B (2x2)
    W shape: (4, 7) for output C (2x2)
    """
    n = m = p = 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]  # 4 rows
    V = [[0] * rank for _ in range(m * p)]  # 4 rows
    W = [[0] * rank for _ in range(p * n)]  # 4 rows

    # P1 = (A00+A11)*(B00+B11)
    # A00 index: 0*2+0=0, A11 index: 1*2+1=3
    # B00 index: 0*2+0=0, B11 index: 1*2+1=3
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    # C00 += P1, C11 += P1
    W[0][0] = 1  # C00 (k*n+i = 0*2+0)
    W[3][0] = 1  # C11 (k*n+i = 1*2+1)

    # P2 = (A10+A11)*B00
    # A10 index: 1*2+0=2, A11 index: 1*2+1=3
    # B00 index: 0*2+0=0
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    # C10 += P2
    W[2][1] = 1  # C10 (k*n+i = 1*2+0)

    # P3 = A00*(B01-B11)
    # A00 index: 0
    # B01 index: 0*2+1=1, B11 index: 1*2+1=3
    U[0][2] = 1  # A00
    V[1][2] = 1  # B01
    V[3][2] = -1  # B11
    # C01 += P3
    W[1][2] = 1  # C01 (k*n+i = 0*2+1)

    # P4 = A11*(B10-B00)
    # A11 index: 3
    # B10 index: 1*2+0=2, B00 index: 0
    U[3][3] = 1  # A11
    V[2][3] = 1  # B10
    V[0][3] = -1  # B00
    # C00 += P4, C10 += P4
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10

    # P5 = (A00+A01)*B11
    # A00 index: 0, A01 index: 0*2+1=1
    # B11 index: 3
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    # C00 -= P5, C01 += P5
    W[0][4] = -1  # C00
    W[1][4] = 1  # C01

    # P6 = (A10-A00)*(B00+B01)
    # A10 index: 2, A00 index: 0
    # B00 index: 0, B01 index: 1
    U[2][5] = 1  # A10
    U[0][5] = -1  # A00
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    # C11 += P6
    W[3][5] = 1  # C11

    # P7 = (A01-A11)*(B10+B11)
    # A01 index: 1, A11 index: 3
    # B10 index: 2, B11 index: 3
    U[1][6] = 1  # A01
    U[3][6] = -1  # A11
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    # C00 += P7
    W[0][6] = 1  # C00

    return U, V, W


def construct():
    """
    Constructs rank-98 decomposition for 4x4 x 4x7 matrix multiplication.
    
    Uses Strassen's algorithm recursively:
    - 4x4x4 multiplication has rank 49 (7x7 from two levels of Strassen)
    - 4x4 x 4x7 splits into two 4x4 x 4x4 multiplications
    - Total rank: 98
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # First, get the rank-49 decomposition for 4x4 x 4x4
    # This uses Strassen at two levels:
    # Level 1: Split 4x4 into four 2x2 blocks
    # Level 2: Each 2x2x2x2x2 uses Strassen rank 7
    
    # For 4x4x4, we have:
    # A is 4x4, B is 4x4, C is 4x4
    # Split into 2x2 blocks of 2x2 submatrices
    # Use Strassen at the block level (7 block products)
    # Each block product is a 2x2 x 2x2 -> 2x2 multiplication (rank 7)
    # Total rank: 7 * 7 = 49
    
    # The block-level Strassen uses indices:
    # Block A: A00, A01, A10, A11 (each 2x2)
    # Block B: B00, B01, B10, B11 (each 2x2)
    # Block C: C00, C01, C10, C11 (each 2x2)
    
    # At the block level, we apply Strassen's 7 products to the blocks
    # Then for each block product, we apply Strassen's 7 products to the 2x2 submatrices
    
    # Let's construct the 4x4x4 decomposition:
    # U shape: (16, 49) for A (4x4)
    # V shape: (16, 49) for B (4x4)  
    # W shape: (16, 49) for C (4x4)
    
    U4 = [[0] * 49 for _ in range(16)]
    V4 = [[0] * 49 for _ in range(16)]
    W4 = [[0] * 49 for _ in range(16)]
    
    # Get the 2x2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2x2()
    
    # Block-level Strassen indices (for the 7 block products)
    # Each block product q (0-6) involves certain blocks of A and B
    # Block indices: 00, 01, 10, 11
    
    # For each block product q, define which blocks of A and B are involved
    # and their coefficients
    
    # P1 = (A00+A11)*(B00+B11): blocks A{00,11} * B{00,11}
    # P2 = (A10+A11)*B00: blocks A{10,11} * B{00}
    # P3 = A00*(B01-B11): blocks A{00} * B{01,-11}
    # P4 = A11*(B10-B00): blocks A{11} * B{10,-00}
    # P5 = (A00+A01)*B11: blocks A{00,01} * B{11}
    # P6 = (A10-A00)*(B00+B01): blocks A{10,-00} * B{00,01}
    # P7 = (A01-A11)*(B10+B11): blocks A{01,-11} * B{10,11}
    
    # For each block product q, and for each sub-product r within that block product:
    # The global rank index is q*7 + r
    
    # Define the block-level coefficients for A and B
    # A_blocks[q] = dict of block_index -> coefficient
    # B_blocks[q] = dict of block_index -> coefficient
    
    A_blocks = [
        {0: 1, 3: 1},      # P1: A00, A11
        {2: 1, 3: 1},      # P2: A10, A11
        {0: 1},            # P3: A00
        {3: 1},            # P4: A11
        {0: 1, 1: 1},      # P5: A00, A01
        {2: 1, 0: -1},     # P6: A10, -A00
        {1: 1, 3: -1},     # P7: A01, -A11
    ]
    
    B_blocks = [
        {0: 1, 3: 1},      # P1: B00, B11
        {0: 1},            # P2: B00
        {1: 1, 3: -1},     # P3: B01, -B11
        {2: 1, 0: -1},     # P4: B10, -B00
        {3: 1},            # P5: B11
        {0: 1, 1: 1},      # P6: B00, B01
        {2: 1, 3: 1},      # P7: B10, B11
    ]
    
    # For each block product q and sub-product r:
    # Global rank index: R = q * 7 + r
    # U4[i, R] where i is the 4x4 index (i_row * 4 + i_col)
    # V4[j, R] where j is the 4x4 index (j_row * 4 + j_col)
    # W4[k, R] where k is the 4x4 index (k_row * 4 + k_col)
    
    # For a given block product q and sub-product r:
    # The block product involves blocks of A and B
    # For each block a_blk in A_blocks[q] and block b_blk in B_blocks[q]:
    # The 2x2 submatrix multiplication uses U2, V2, W2
    # The global indices are:
    #   A index: a_blk_row * 2 + sub_row, a_blk_col * 2 + sub_col
    #   B index: b_blk_row * 2 + sub_row2, b_blk_col * 2 + sub_col2
    #   C index: c_blk_row * 2 + sub_row3, c_blk_col * 2 + sub_col3
    
    # But we need to be careful: the block-level Strassen combines blocks
    # For block product q, the contribution to C comes from specific C blocks
    
    # Let's define C_blocks[q] = dict of block_index -> coefficient
    C_blocks = [
        {0: 1, 3: 1},      # P1: C00 += P1, C11 += P1
        {2: 1},            # P2: C10 += P2
        {1: 1},            # P3: C01 += P3
        {0: 1, 2: 1},      # P4: C00 += P4, C10 += P4
        {0: -1, 1: 1},     # P5: C00 -= P5, C01 += P5
        {3: 1},            # P6: C11 += P6
        {0: 1},            # P7: C00 += P7
    ]
    
    for q in range(7):
        for r in range(7):
            R = q * 7 + r
            
            # Get the coefficients for A blocks, B blocks, C blocks
            a_coeff = A_blocks[q]
            b_coeff = B_blocks[q]
            c_coeff = C_blocks[q]
            
            # For the 2x2x2 sub-problem (index r), we use U2[*, r], V2[*, r], W2[*, r]
            # But we need to map the block indices to global indices
            
            # For each combination of A block, B block, C block involved in this product:
            # Actually, the Strassen product is a single scalar product of two vectors
            # The block-level Strassen says:
            # Product_q = (sum of A blocks with coeffs) * (sum of B blocks with coeffs)
            # And C gets contributions from Product_q to specific C blocks
            
            # But since we're using rank decomposition, each "rank term" corresponds to
            # a product of a linear form in A, a linear form in B, and a linear form in C
            
            # For the 4x4x4 case using Strassen twice:
            # The overall rank term (q, r) corresponds to:
            # A_linear_form = sum over a_blk in A_blocks[q] of coeff_a(a_blk) * A2[a_blk, r]
            # B_linear_form = sum over b_blk in B_blocks[q] of coeff_b(b_blk) * B2[b_blk, r]
            # C_linear_form = sum over c_blk in C_blocks[q] of coeff_c(c_blk) * C2[c_blk, r]
            
            # Wait, this isn't quite right. Let me reconsider.
            
            # Actually, the way to compose two Strassen decompositions:
            # At the block level, we have 7 products P_q, each of which is a product of 
            # linear combinations of 2x2 blocks.
            # Each P_q is itself computed using a rank-7 decomposition for 2x2x2 multiplication.
            
            # So for block product q, and sub-product r:
            # The A contribution is: for each block a_blk with coefficient ca in A_blocks[q],
            #   the sub-block A2[a_blk] contributes ca * U2[local_a_idx, r]
            # Similarly for B and C.
            
            # Let's implement this:
            
            # For A: iterate over blocks in A_blocks[q]
            for a_blk, ca in a_coeff.items():
                a_blk_row = a_blk // 2  # 0 or 1
                a_blk_col = a_blk % 2   # 0 or 1
                
                # For each element in the 2x2 sub-block (using U2[*, r]):
                for sub_i in range(2):
                    for sub_j in range(2):
                        local_idx = sub_i * 2 + sub_j
                        u_val = U2[local_idx][r]
                        if u_val != 0:
                            global_i = a_blk_row * 2 + sub_i
                            global_j = a_blk_col * 2 + sub_j
                            global_idx = global_i * 4 + global_j
                            U4[global_idx][R] += ca * u_val
            
            # For B: iterate over blocks in B_blocks[q]
            for b_blk, cb in b_coeff.items():
                b_blk_row = b_blk // 2
                b_blk_col = b_blk % 2
                
                for sub_j in range(2):
                    for sub_k in range(2):
                        local_idx = sub_j * 2 + sub_k
                        v_val = V2[local_idx][r]
                        if v_val != 0:
                            global_j = b_blk_row * 2 + sub_j
                            global_k = b_blk_col * 2 + sub_k
                            global_idx = global_j * 4 + global_k
                            V4[global_idx][R] += cb * v_val
            
            # For C: iterate over blocks in C_blocks[q]
            for c_blk, cc in c_coeff.items():
                c_blk_row = c_blk // 2
                c_blk_col = c_blk % 2
                
                for sub_k in range(2):
                    for sub_i in range(2):
                        local_idx = sub_k * 2 + sub_i
                        w_val = W2[local_idx][r]
                        if w_val != 0:
                            global_k = c_blk_row * 2 + sub_k
                            global_i = c_blk_col * 2 + sub_i
                            global_idx = global_k * 4 + global_i
                            W4[global_idx][R] += cc * w_val
    
    # Now we have U4, V4, W4 for 4x4 x 4x4 -> 4x4, rank 49
    # U4 shape: (16, 49)
    # V4 shape: (16, 49)
    # W4 shape: (16, 49)
    
    # For 4x4 x 4x7:
    # Split B (4x7) into B1 (4x4) and B2 (4x3)
    # Treat B2 as 4x4 with last column zero
    # C = A*B1 + A*B2
    # C1 = A*B1 (4x4), C2 = A*B2 (4x3)
    # C = [C1 | C2] (4x7)
    
    # Total rank: 49 + 49 = 98
    # U shape: (16, 98)
    # V shape: (28, 98)  # 4*7 = 28
    # W shape: (28, 98)  # 7*4 = 28
    
    U = [[0] * 98 for _ in range(16)]
    V = [[0] * 98 for _ in range(28)]
    W = [[0] * 98 for _ in range(28)]
    
    # Part 1: B1 is columns 0-3 of B, C1 is columns 0-3 of C
    # Use ranks 0-48
    for i in range(16):
        for R in range(49):
            U[i][R] = U4[i][R]
    
    for j in range(4):
        for k in range(4):
            v_idx = j * 7 + k
            for R in range(49):
                V[v_idx][R] = V4[j * 4 + k][R]
    
    for k in range(4):
        for i in range(4):
            w_idx = k * 4 + i
            for R in range(49):
                W[w_idx][R] = W4[k * 4 + i][R]
    
    # Part 2: B2 is columns 4-6 of B (treat as 4x4 with col 3 zero), C2 is columns 4-6 of C
    # Use ranks 49-97
    for i in range(16):
        for R in range(49):
            U[i][R + 49] = U4[i][R]
    
    for j in range(4):
        for k in range(3):  # Only first 3 columns
            v_idx = j * 7 + (4 + k)
            v4_idx = j * 4 + k
            for R in range(49):
                V[v_idx][R + 49] = V4[v4_idx][R]
        # Column k=3 is zero, so V stays 0 for those
    
    for k in range(3):  # Only first 3 columns
        for i in range(4):
            w_idx = (4 + k) * 4 + i
            w4_idx = k * 4 + i
            for R in range(49):
                W[w_idx][R + 49] = W4[w4_idx][R]
    # Columns k=3,4,5,6,7 of W are zero (for C2, only cols 0-2 are non-zero)
    # Wait, C2 is 4x3, so k ranges 0-2 in local, which maps to k=4-6 in global
    # So W indices for k=3,7 are zero, and for k=4,5,6 we use k_local=0,1,2
    
    # Let me fix: C is 4x7, so k ranges 0-6
    # C1 = cols 0-3, C2 = cols 4-6
    # For Part 2, local k=0,1,2 maps to global k=4,5,6
    
    # Already handled above correctly
    
    return {"U": U, "V": V, "W": W}
