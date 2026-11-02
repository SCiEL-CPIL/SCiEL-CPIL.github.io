"""Matrix multiplication 6x6x6 using Strassen's algorithm on 2x2 blocks."""

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    t_start = time.time()
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # We'll use Strassen's algorithm on 2x2 blocks.
    # 6x6 = 3x3 grid of 2x2 blocks.
    # Total rank = 3*3*3 * 7 = 189 (for each output block (i,k), sum over j, each 2x2 mult costs 7)
    
    # Actually, let me think more carefully.
    # C[i][k] = sum_j A[i][j] * B[j][k] for block indices i,j,k in 0..2
    # Each 2x2 multiplication uses Strassen with 7 products.
    # For fixed (i,k), we have 3 j values, each contributing 7 rank-1 terms.
    # So 21 terms per (i,k), and 9 (i,k) pairs = 189 total.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    # Strassen's 7 products for 2x2 multiplication:
    # Given X = [[x00, x01], [x10, x11]] and Y = [[y00, y01], [y10, y11]]
    # P1 = (x00+x11)*(y00+y11)
    # P2 = (x10+x11)*y00
    # P3 = x00*(y01-y11)
    # P4 = x11*(y10-y00)
    # P5 = (x00+x01)*y11
    # P6 = (x10-x00)*(y00+y10)
    # P7 = (x01-x11)*(y10+y11)
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # For each (i, j, k) in 0..2:
    #   A block A[i][j] has global indices: rows 2i, 2i+1; cols 2j, 2j+1
    #   B block B[j][k] has global indices: rows 2j, 2j+1; cols 2k, 2k+1
    #   Product contributes to C[i][k]: rows 2i, 2i+1; cols 2k, 2k+1
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # A block indices: (2i, 2j), (2i, 2j+1), (2i+1, 2j), (2i+1, 2j+1)
                # B block indices: (2j, 2k), (2j, 2k+1), (2j+1, 2k), (2j+1, 2k+1)
                
                # Strassen products for this (i,j,k) combination
                # Each product is a rank-1 term in the tensor decomposition.
                
                # Let's define the 7 Strassen terms:
                # P1 = (A00+A11)(B00+B11)
                # P2 = (A10+A11)B00
                # P3 = A00(B01-B11)
                # P4 = A11(B10-B00)
                # P5 = (A00+A01)B11
                # P6 = (A10-A00)(B00+B10)
                # P7 = (A01-A11)(B10+B11)
                
                # For P1: (A00+A11)(B00+B11)
                # A part: A00 + A11 => U entries at (2i, 2j)=1, (2i+1, 2j+1)=1
                # B part: B00 + B11 => V entries at (2j, 2k)=1, (2j+1, 2k+1)=1
                # C part: contributes to Z00 and Z11 with coefficient +1
                # Z00 is C[2i][2k], Z11 is C[2i+1][2k+1]
                
                def add_term(a_entries, b_entries, c_entries):
                    """Add a rank-1 term. a_entries, b_entries, c_entries are dicts {index: coeff}"""
                    nonlocal col
                    for idx, coeff in a_entries.items():
                        U[idx][col] = coeff
                    for idx, coeff in b_entries.items():
                        V[idx][col] = coeff
                    for idx, coeff in c_entries.items():
                        W[idx][col] = coeff
                    col += 1
                
                # P1 = (A00+A11)(B00+B11)
                # Contributes: Z00 += P1, Z11 += P1
                a1 = {(2*i, 2*j): 1, (2*i+1, 2*j+1): 1}
                b1 = {(2*j, 2*k): 1, (2*j+1, 2*k+1): 1}
                c1 = {(2*k, 2*i): 1, (2*k+1, 2*i+1): 1}  # W index is k*n+i
                # But we need to add this as one term with combined coefficients
                # Actually, each rank-1 term has a single product. Let me reconsider.
                
                # The tensor T[i,j,k] for matrix mult C_ik = sum_j A_ij B_jk
                # A rank-1 decomposition term is: for each product, we have
                # U[row_a, col] = coeff_a, V[row_b, col] = coeff_b, W[row_c, col] = coeff_c
                # such that sum_col U[row_a,col]*V[row_b,col]*W[row_c,col] gives C[row_c_a, row_c_b]
                
                # For P1 = (A[2i,2j] + A[2i+1,2j+1]) * (B[2j,2k] + B[2j+1,2k+1])
                # This expands to 4 terms, but in tensor decomposition we keep it as one rank-1 term
                # with the sums encoded in U and V.
                
                # Wait, I need to be more careful. In a rank-1 tensor decomposition,
                # each term is a product of three vectors. The Strassen algorithm
                # uses 7 such products per 2x2 multiplication.
                
                # Let me redefine. For the 2x2 block multiplication at (i,j,k):
                # The 7 Strassen products are:
                # P1 = (x00+x11)*(y00+y11) where x is A[i][j] block, y is B[j][k] block
                # In the global tensor, this is one rank-1 term where:
                # U has 1s at positions corresponding to x00 and x11
                # V has 1s at positions corresponding to y00 and y11
                # W has 1s at positions where this product contributes to C
                
                # For P1: contributes to Z00 (C[2i][2k]) and Z11 (C[2i+1][2k+1])
                # So W should have 1 at (2k, 2i) and 1 at (2k+1, 2i+1)
                
                # But wait - this means the W vector for this term has two non-zero entries.
                # That's fine - it's still a valid rank-1 term in the tensor decomposition.
                
                # Let me redo this properly:
                
                # P1: (A[2i,2j] + A[2i+1,2j+1]) * (B[2j,2k] + B[2j+1,2k+1])
                # => C[2i,2k] += P1, C[2i+1,2k+1] += P1
                add_term(
                    {2*i*m + 2*j: 1, (2*i+1)*m + (2*j+1): 1},  # U indices
                    {2*j*p + 2*k: 1, (2*j+1)*p + (2*k+1): 1},  # V indices
                    {2*k*n + 2*i: 1, (2*k+1)*n + (2*i+1): 1}   # W indices
                )
                
                # P2: (A[2i+1,2j] + A[2i+1,2j+1]) * B[2j,2k]
                # => C[2i+1,2k] += P2
                add_term(
                    {(2*i+1)*m + 2*j: 1, (2*i+1)*m + (2*j+1): 1},
                    {2*j*p + 2*k: 1},
                    {(2*k+1)*n + 2*i: 1}  # W index for C[2i+1][2k] is (2k)*n + (2i+1)
                )
                
                # P3: A[2i,2j] * (B[2j,2k+1] - B[2j+1,2k+1])
                # => C[2i,2k+1] += P3
                add_term(
                    {2*i*m + 2*j: 1},
                    {2*j*p + (2*k+1): 1, (2*j+1)*p + (2*k+1): -1},
                    {2*k*n + (2*i+1): 1}  # W index for C[2i+1... wait
                )
                # C[2i][2k+1]: W index = (2k+1)*n + 2i
                # Let me fix: W index is k_global * n + i_global
                # For C[row_c][col_c], W index = col_c * n + row_c
                # C[2i][2k+1] => W index = (2k+1)*n + 2i
                # I made an error above. Let me redo.
                
                # Let me restart the Strassen terms more carefully.
                # I'll reset and do it correctly.
                
                # Actually, I already added 3 terms incorrectly. Let me just build the whole thing fresh.
                
                break
            break
        break
    break
    
    # Let me rebuild from scratch with correct indexing.
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # A block: rows 2i,2i+1; cols 2j,2j+1
                # B block: rows 2j,2j+1; cols 2k,2k+1
                # C block: rows 2i,2i+1; cols 2k,2k+1
                
                # Strassen 7 products:
                
                # P1 = (x00+x11)*(y00+y11)
                # Contributes to z00 and z11
                U[2*i*m + 2*j][col] = 1
                U[(2*i+1)*m + (2*j+1)][col] = 1
                V[2*j*p + 2*k][col] = 1
                V[(2*j+1)*p + (2*k+1)][col] = 1
                W[2*k*n + 2*i][col] = 1
                W[(2*k+1)*n + (2*i+1)][col] = 1
                col += 1
                
                # P2 = (x10+x11)*y00
                # Contributes to z10
                U[(2*i+1)*m + 2*j][col] = 1
                U[(2*i+1)*m + (2*j+1)][col] = 1
                V[2*j*p + 2*k][col] = 1
                W[2*k*n + (2*i+1)][col] = 1
                col += 1
                
                # P3 = x00*(y01-y11)
                # Contributes to z01
                U[2*i*m + 2*j][col] = 1
                V[2*j*p + (2*k+1)][col] = 1
                V[(2*j+1)*p + (2*k+1)][col] = -1
                W[(2*k+1)*n + 2*i][col] = 1
                col += 1
                
                # P4 = x11*(y10-y00)
                # Contributes to z00 and z10
                U[(2*i+1)*m + (2*j+1)][col] = 1
                V[(2*j+1)*p + 2*k][col] = 1
                V[2*j*p + 2*k][col] = -1
                W[2*k*n + 2*i][col] = 1
                W[2*k*n + (2*i+1)][col] = 1
                col += 1
                
                # P5 = (x00+x01)*y11
                # Contributes to z00 and z01
                U[2*i*m + 2*j][col] = 1
                U[2*i*m + (2*j+1)][col] = 1
                V[(2*j+1)*p + (2*k+1)][col] = 1
                W[2*k*n + 2*i][col] = -1
                W[(2*k+1)*n + 2*i][col] = 1
                col += 1
                
                # P6 = (x10-x00)*(y00+y10)
                # Contributes to z11
                U[(2*i+1)*m + 2*j][col] = 1
                U[2*i*m + 2*j][col] = -1
                V[2*j*p + 2*k][col] = 1
                V[(2*j+1)*p + 2*k][col] = 1
                W[(2*k+1)*n + (2*i+1)][col] = 1
                col += 1
                
                # P7 = (x01-x11)*(y10+y11)
                # Contributes to z00 and z11
                U[2*i*m + (2*j+1)][col] = 1
                U[(2*i+1)*m + (2*j+1)][col] = -1
                V[(2*j+1)*p + 2*k][col] = 1
                V[(2*j+1)*p + (2*k+1)][col] = 1
                W[2*k*n + 2*i][col] = 1
                W[(2*k+1)*n + (2*i+1)][col] = -1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

