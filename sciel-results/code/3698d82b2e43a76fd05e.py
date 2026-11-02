"""Matrix multiplication 6x6x6 using Strassen algorithm for 2x2 blocks of 3x3 matrices."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 6, 6, 6
    # We use Strassen for 2x2 blocks of size 3x3.
    # This requires 7 multiplications of 3x3 matrices.
    # Each 3x3 multiplication is done with schoolbook method (rank 27).
    # Total rank = 7 * 27 = 189.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a bilinear term to U, V, W
    # u_coeffs: list of 9 coeffs for A block (3x3)
    # v_coeffs: list of 9 coeffs for B block (3x3)
    # w_target: (block_row, block_col, local_row, local_col) and coeff
    # Actually, it's easier to just expand the linear forms.
    
    # Let's define the 7 Strassen products.
    # Each product is X * Y where X and Y are 3x3 matrices formed from blocks.
    # Blocks of A: A00, A01, A10, A11
    # Blocks of B: B00, B01, B10, B11
    
    # We represent a 3x3 matrix X as a 3x3 array of lists of (block, coeff)
    # Or simpler: X[i][j] is a list of (block_idx, coeff)
    # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
    
    def make_block_matrix(coeffs_dict):
        # coeffs_dict: {block_idx: 3x3 matrix of coeffs}
        # Returns 3x3 list of lists of (block_idx, coeff)
        M = [[[] for _ in range(3)] for _ in range(3)]
        for b_idx, mat in coeffs_dict.items():
            for i in range(3):
                for j in range(3):
                    if mat[i][j] != 0:
                        M[i][j].append((b_idx, mat[i][j]))
        return M

    # Identity 3x3
    I = [[0]*3 for _ in range(3)]
    I[0][0] = 1; I[1][1] = 1; I[2][2] = 1
    
    # Define the 7 X and Y matrices
    # M1 = (A00+A11)(B00+B11)
    X1 = make_block_matrix({0: I, 3: I})
    Y1 = make_block_matrix({0: I, 3: I})
    
    # M2 = (A10+A11)B00
    X2 = make_block_matrix({2: I, 3: I})
    Y2 = make_block_matrix({0: I})
    
    # M3 = A00(B10-B11)
    X3 = make_block_matrix({0: I})
    Y3 = make_block_matrix({2: I, 3: [[-1]*3 for _ in range(3)]}) # B10 - B11
    
    # M4 = A11(B01-B00)
    X4 = make_block_matrix({3: I})
    Y4 = make_block_matrix({1: I, 0: [[-1]*3 for _ in range(3)]}) # B01 - B00
    
    # M5 = (A00+A01)B11
    X5 = make_block_matrix({0: I, 1: I})
    Y5 = make_block_matrix({3: I})
    
    # M6 = (A10-A00)(B00+B01)
    X6 = make_block_matrix({2: I, 0: [[-1]*3 for _ in range(3)]}) # A10 - A00
    Y6 = make_block_matrix({0: I, 1: I})
    
    # M7 = (A01-A11)(B10+B11)
    X7 = make_block_matrix({1: I, 3: [[-1]*3 for _ in range(3)]}) # A01 - A11
    Y7 = make_block_matrix({2: I, 3: I})
    
    products = [
        (X1, Y1, [(0,0,1), (1,1,1)]), # M1 contributes to C00 (+1) and C11 (+1)
        (X2, Y2, [(1,0,1)]),          # M2 contributes to C10 (+1)
        (X3, Y3, [(0,1,1)]),          # M3 contributes to C01 (+1)
        (X4, Y4, [(0,0,1), (1,0,1)]), # M4 contributes to C00 (+1) and C10 (+1)
        (X5, Y5, [(0,0,-1), (0,1,1)]),# M5 contributes to C00 (-1) and C01 (+1)
        (X6, Y6, [(1,1,1)]),          # M6 contributes to C11 (+1)
        (X7, Y7, [(0,0,1), (1,1,-1)]) # M7 contributes to C00 (+1) and C11 (-1)
    ]
    
    # Block offsets for A (input 1)
    # A00: 0, A01: 3, A10: 9, A11: 12 (row-major 6x6)
    # Actually, U is indexed by i*m+j.
    # A00 is rows 0-2, cols 0-2. Indices 0..2, 3..5, 6..8? No.
    # A is 6x6.
    # A[0][0] is index 0.
    # A[0][3] is index 3.
    # A[3][0] is index 18.
    # A[3][3] is index 21.
    
    a_block_offsets = [0, 3, 18, 21] # A00, A01, A10, A11
    b_block_offsets = [0, 3, 18, 21] # B00, B01, B10, B11
    c_block_offsets = [0, 3, 18, 21] # C00, C01, C10, C11
    
    for X, Y, c_contribs in products:
        for i in range(3):
            for j in range(3):
                # Compute Z[i][j] = sum_k X[i][k] * Y[k][j]
                for k in range(3):
                    # X[i][k] is a list of (block, coeff)
                    # Y[k][j] is a list of (block, coeff)
                    # We need to multiply the linear forms.
                    # L_A = sum_{(b, c)} c * A_block[b][i][k]
                    # L_B = sum_{(b, c)} c * B_block[b][k][j]
                    # Term = L_A * L_B
                    # This expands to sum_{b1, b2} c1 c2 A[b1][i][k] B[b2][k][j]
                    
                    for b1, c1 in X[i][k]:
                        for b2, c2 in Y[k][j]:
                            coeff = c1 * c2
                            if coeff == 0: continue
                            
                            # Add to U, V, W
                            # U index: a_block_offsets[b1] + i*3 + k
                            # V index: b_block_offsets[b2] + k*3 + j
                            
                            u_idx = a_block_offsets[b1] + i * 3 + k
                            v_idx = b_block_offsets[b2] + k * 3 + j
                            
                            U[u_idx][col] += coeff
                            V[v_idx][col] += coeff
                            
                            # Update W for all C blocks that this M contributes to
                            for c_block, c_coeff in c_contribs:
                                # C block index: c_block_offsets[c_block] + j*3 + i?
                                # C is 6x6.
                                # C00 is rows 0-2, cols 0-2.
                                # The entry (i,j) in the 3x3 block corresponds to (i,j) in C00.
                                # Index in W: c_block_offsets[c_block] + i*3 + j?
                                # Wait, W index is k*n+i?
                                # Task says: W[k*n+i, r].
                                # C[i][k] = sum ...
                                # My loop variables: i is row of C block, j is col of C block.
                                # So C entry is (i, j).
                                # W index should correspond to C entry (i, j).
                                # W index = j*n + i?
                                # Let's check the convention: W[k*n+i, r] where C[i,k] is the output.
                                # In my loop, i is row, j is col. So C[i][j].
                                # So W index = j*6 + i.
                                
                                w_idx = c_block_offsets[c_block] + j * 3 + i
                                # Wait, c_block_offsets gives the top-left corner.
                                # If c_block=0 (C00), offset=0. Entry (i,j) is at 0 + i*3 + j?
                                # C is stored row-major?
                                # U is i*m+j.
                                # V is j*p+k.
                                # W is k*n+i.
                                # So C[i][k] is at index k*n+i.
                                # Here i is row, j is col (which is k in the formula C[i][k]).
                                # So W index = j*6 + i.
                                # But we are inside a block.
                                # Global row = c_block_row * 3 + i.
                                # Global col = c_block_col * 3 + j.
                                # W index = (Global Col) * 6 + (Global Row).
                                
                                global_row = (c_block // 2) * 3 + i
                                global_col = (c_block % 2) * 3 + j
                                w_idx_global = global_col * 6 + global_row
                                
                                W[w_idx_global][col] += coeff * c_coeff
                            
                            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
