"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 3 (blocks) * 3 (blocks) * 3 (sum) * 7 (Strassen) = 189
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # P1: (A00+A11)(B00+B11)
    # P2: (A10+A11)B00
    # P3: A00(B01-B11)
    # P4: A11(B10-B00)
    # P5: (A00+A01)B11
    # P6: (A10-A00)(B00+B01)
    # P7: (A01-A11)(B10+B11)
    
    # (u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w)
    # Indices are (row, col) for U, (row, col) for V, (row, col) for W
    strassen_terms = [
        # P1
        [(0, 0, 1), (0, 0, 1), (0, 0, 1)], # A00*B00 part of P1? No, P1 is one product.
        # We need to represent P1 as a sum of rank-1 tensors? No, P1 is a scalar product of sums.
        # P1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # This is not a single rank-1 tensor.
        # Strassen's algorithm replaces 8 products with 7.
        # The 7 products are:
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B01-B11)
        # P4 = A11(B10-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        #
        # Each Pk is a scalar.
        # In the tensor decomposition, each Pk corresponds to a set of rank-1 terms?
        # No, the decomposition is:
        # C = sum_r (sum_i u_ri A_i) (sum_j v_rj B_j) w_r
        # For Strassen, each "product" Pk is formed by linear combinations of A and B entries.
        # So for P1:
        # u_1 = A00 + A11
        # v_1 = B00 + B11
        # This is not a rank-1 tensor in the standard sense because u_1 is a sum.
        # However, the tensor decomposition allows linear combinations in U and V.
        # So for P1, we have:
        # U[0,0] += 1, U[1,1] += 1
        # V[0,0] += 1, V[1,1] += 1
        # W[0,0] += 1, W[1,1] += 1  (because C00 and C11 use P1)
        # Wait, C00 = P1 + P4 - P5 + P7
        # C11 = P1 - P2 + P3 + P6
        # So P1 contributes to W[0,0] and W[1,1].
        #
        # Let's list the contributions for each Pk to U, V, W.
    ]
    
    # We will iterate over each block (I, K, J) and each Strassen product (0..6)
    col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                # Base indices
                u_row_base = 2 * I
                u_col_base = 2 * K
                v_row_base = 2 * K
                v_col_base = 2 * J
                w_row_base = 2 * I
                w_col_base = 2 * J
                
                # P1
                U[u_row_base + 0][col] = 1
                U[u_row_base + 1][col] = 1
                V[v_row_base + 0][col] = 1
                V[v_row_base + 1][col] = 1
                W[w_row_base + 0][col] = 1
                W[w_row_base + 1][col] = 1
                col += 1
                
                # P2
                U[u_row_base + 1][col] = 1
                U[u_row_base + 1][col] = 1 # A10 + A11? No, P2 = (A10+A11)B00
                # A10 is (1,0), A11 is (1,1)
                U[u_row_base + 1][col] = 1
                U[u_row_base + 1][col] = 1 # This is wrong. U[1,0] and U[1,1]
                # Let's be precise.
                # P2: u = A10 + A11. v = B00.
                # U[1,0] += 1, U[1,1] += 1. V[0,0] += 1.
                # W: P2 contributes to C10. W[1,0] += 1.
                # I already incremented col. Let's reset and do it properly.
                
    # Let's restart the loop logic with a helper function for Strassen terms.
    
    def add_strassen_block(U, V, W, I, K, J, col_offset):
        u_r = 2 * I
        u_c = 2 * K
        v_r = 2 * K
        v_c = 2 * J
        w_r = 2 * I
        w_c = 2 * J
        
        # P1: (A00+A11)(B00+B11)
        U[u_r+0][col_offset+0] = 1
        U[u_r+1][col_offset+0] = 1
        V[v_r+0][col_offset+0] = 1
        V[v_r+1][col_offset+0] = 1
        W[w_r+0][col_offset+0] = 1
        W[w_r+1][col_offset+0] = 1
        
        # P2: (A10+A11)B00
        U[u_r+1][col_offset+1] = 1
        U[u_r+1][col_offset+1] = 1 # A10 is (1,0), A11 is (1,1). So U[1,0] and U[1,1].
        # Wait, U indices are (row, col).
        # A10 is row 1, col 0. A11 is row 1, col 1.
        U[u_r+1][col_offset+1] = 1 # This is U[1,1]. We need U[1,0] too.
        # I am overwriting. I need to add to existing values or use unique columns.
        # Each Pk uses a unique column.
        # So for P2, column is col_offset+1.
        # U[u_r+1, u_c+0] = 1
        # U[u_r+1, u_c+1] = 1
        # V[v_r+0, v_c+0] = 1
        # W[w_r+1, w_c+0] = 1
        
        # Let's write it explicitly for each term.
        
        # P1 (col 0)
        U[u_r+0][col_offset+0] += 1
        U[u_r+1][col_offset+0] += 1
        V[v_r+0][col_offset+0] += 1
        V[v_r+1][col_offset+0] += 1
        W[w_r+0][col_offset+0] += 1
        W[w_r+1][col_offset+0] += 1
        
        # P2 (col 1)
        U[u_r+1][col_offset+1] += 1 # A10
        U[u_r+1][col_offset+1] += 1 # A11? No, U[1,0] and U[1,1].
        # U[row, col].
        # A10 is (1,0). A11 is (1,1).
        # So U[1,0] and U[1,1].
        # My indices: u_r+1 is row 1. u_c+0 is col 0. u_c+1 is col 1.
        # So U[u_r+1, u_c+0] and U[u_r+1, u_c+1].
        # I wrote U[u_r+1][col_offset+1]. This is row 1, col (col_offset+1).
        # But col_offset is the column index in the decomposition, not the matrix column.
        # The matrix column is u_c + local_col.
        # So:
        U[u_r+1][col_offset+1] = 1 # This is wrong. This sets the decomposition column.
        # I need to set U[local_row, local_col] for the decomposition column.
        # U is a 2D array: U[global_row, decomp_col].
        # So U[global_row] is a list.
        # I need to add 1 to U[global_row][decomp_col].
        
        # Correct way:
        # P2:
        U[u_r+1][col_offset+1] += 1 # Row 1, Decomp Col 1. But what about Matrix Col?
        # The matrix col is determined by the local index.
        # But U only has one column index (the decomposition column).
        # The row index of U encodes the matrix row AND column?
        # No, U has shape (n*m, rank).
        # So U[row_index, decomp_col].
        # row_index = i*m + j.
        # So for A10, row_index = 1*m + 0.
        # For A11, row_index = 1*m + 1.
        # So I need to set U[1*m+0][col] and U[1*m+1][col].
        
        # Let's redefine the helper to be safe.
        return col_offset + 7

    # Re-implement with correct indexing.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                # Local to Global mapping
                # U row: I*m + K? No, U row is i*m + j.
                # Here i=I (block), j=K (block). But within block, we have 2x2.
                # So global row = (2*I + alpha)*m + (2*K + beta).
                # m=6.
                # So global row = (2I + alpha)*6 + (2K + beta).
                
                # V row: (2*K + beta)*6 + (2*J + delta).
                # W row: (2*I + alpha)*6 + (2*J + delta).
                
                # P1: (A00+A11)(B00+B11)
                # A00: alpha=0, beta=0. A11: alpha=1, beta=1.
                # B00: beta=0, delta=0. B11: beta=1, delta=1.
                # C00: alpha=0, delta=0. C11: alpha=1, delta=1.
                
                # U contributions
                U[(2*I+0)*6 + (2*K+0)][col] = 1
                U[(2*I+1)*6 + (2*K+1)][col] = 1
                # V contributions
                V[(2*K+0)*6 + (2*J+0)][col] = 1
                V[(2*K+1)*6 + (2*J+1)][col] = 1
                # W contributions
                W[(2*I+0)*6 + (2*J+0)][col] = 1
                W[(2*I+1)*6 + (2*J+1)][col] = 1
                col += 1
                
                # P2: (A10+A11)B00
                # A10: alpha=1, beta=0. A11: alpha=1, beta=1.
                # B00: beta=0, delta=0.
                # C10: alpha=1, delta=0.
                
                U[(2*I+1)*6 + (2*K+0)][col] = 1
                U[(2*I+1)*6 + (2*K+1)][col] = 1
                V[(2*K+0)*6 + (2*J+0)][col] = 1
                W[(2*I+1)*6 + (2*J+0)][col] = 1
                col += 1
                
                # P3: A00(B01-B11)
                # A00: alpha=0, beta=0.
                # B01: beta=0, delta=1. B11: beta=1, delta=1.
                # C01: alpha=0, delta=1. C11: alpha=1, delta=1.
                
                U[(2*I+0)*6 + (2*K+0)][col] = 1
                V[(2*K+0)*6 + (2*J+1)][col] = 1
                V[(2*K+1)*6 + (2*J+1)][col] = -1
                W[(2*I+0)*6 + (2*J+1)][col] = 1
                W[(2*I+1)*6 + (2*J+1)][col] = 1
                col += 1
                
                # P4: A11(B10-B00)
                # A11: alpha=1, beta=1.
                # B10: beta=1, delta=0. B00: beta=0, delta=0.
                # C00: alpha=0, delta=0. C10: alpha=1, delta=0.
                
                U[(2*I+1)*6 + (2*K+1)][col] = 1
                V[(2*K+1)*6 + (2*J+0)][col] = 1
                V[(2*K+0)*6 + (2*J+0)][col] = -1
                W[(2*I+0)*6 + (2*J+0)][col] = 1
                W[(2*I+1)*6 + (2*J+0)][col] = 1
                col += 1
                
                # P5: (A00+A01)B11
                # A00: alpha=0, beta=0. A01: alpha=0, beta=1.
                # B11: beta=1, delta=1.
                # C01: alpha=0, delta=1. C11: alpha=1, delta=1.
                
                U[(2*I+0)*6 + (2*K+0)][col] = 1
                U[(2*I+0)*6 + (2*K+1)][col] = 1
                V[(2*K+1)*6 + (2*J+1)][col] = 1
                W[(2*I+0)*6 + (2*J+1)][col] = 1
                W[(2*I+1)*6 + (2*J+1)][col] = -1
                col += 1
                
                # P6: (A10-A00)(B00+B01)
                # A10: alpha=1, beta=0. A00: alpha=0, beta=0.
                # B00: beta=0, delta=0. B01: beta=0, delta=1.
                # C11: alpha=1, delta=1.
                
                U[(2*I+1)*6 + (2*K+0)][col] = 1
                U[(2*I+0)*6 + (2*K+0)][col] = -1
                V[(2*K+0)*6 + (2*J+0)][col] = 1
                V[(2*K+0)*6 + (2*J+1)][col] = 1
                W[(2*I+1)*6 + (2*J+1)][col] = 1
                col += 1
                
                # P7: (A01-A11)(B10+B11)
                # A01: alpha=0, beta=1. A11: alpha=1, beta=1.
                # B10: beta=1, delta=0. B11: beta=1, delta=1.
                # C00: alpha=0, delta=0. C11: alpha=1, delta=1.
                
                U[(2*I+0)*6 + (2*K+1)][col] = 1
                U[(2*I+1)*6 + (2*K+1)][col] = -1
                V[(2*K+1)*6 + (2*J+0)][col] = 1
                V[(2*K+1)*6 + (2*J+1)][col] = 1
                W[(2*I+0)*6 + (2*J+0)][col] = 1
                W[(2*I+1)*6 + (2*J+1)][col] = -1
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
