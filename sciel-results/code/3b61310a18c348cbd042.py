import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 189 via Block-Strassen (7 * 27)
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    # u_idx: row in U (A indices)
    # v_idx: row in V (B indices)
    # w_idx: row in W (C indices)
    # u_val, v_val, w_val: coefficients
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if u_val == 0 or v_val == 0 or w_val == 0:
            return
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Strassen 7 products for 3x3 blocks
    # A blocks: A11 (0:3, 0:3), A12 (0:3, 3:6), A21 (3:6, 0:3), A22 (3:6, 3:6)
    # B blocks: B11 (0:3, 0:3), B12 (0:3, 3:6), B21 (3:6, 0:3), B22 (3:6, 3:6)
    
    # Define the 7 products M1..M7
    # Each product is of the form (Linear Comb of A blocks) * (Linear Comb of B blocks)
    # We represent the linear combos as lists of (block_id, coeff)
    
    # Block IDs: 0=A11, 1=A12, 2=A21, 3=A22
    # Block IDs for B: 0=B11, 1=B12, 2=B21, 3=B22
    
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    products = [
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]), # M1
        ([ (2, 1), (3, 1) ], [ (0, 1) ]),          # M2
        ([ (0, 1) ], [ (1, 1), (3, -1) ]),         # M3
        ([ (3, 1) ], [ (2, 1), (0, -1) ]),         # M4
        ([ (0, 1), (1, 1) ], [ (3, 1) ]),          # M5
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ]), # M6
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ])  # M7
    ]
    
    # For each product, we perform schoolbook 3x3 multiplication
    # The result of each product is a 3x3 matrix C_block
    # C_block[i, k] = sum_j X[i, j] Y[j, k]
    
    # We need to map block-local indices to global indices
    # Block (br, bc) where br, bc in {0, 1}
    # Global row for A: br*3 + i
    # Global col for A: bc*3 + j
    # Global row for B: br*3 + j
    # Global col for B: bc*3 + k
    # Global row for C: br*3 + i
    # Global col for C: bc*3 + k
    
    # Wait, the Strassen formula gives the final C blocks.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # This means we can't just add terms for M1..M7 directly to the final C.
    # We must add terms to U, V, W such that the sum of contributions equals the final C.
    # Since the decomposition is linear, we can treat each term in the expansion of C as a separate rank-1 component.
    
    # Expansion of C11[i, k]:
    # sum_j ( (A11[i,j]+A22[i,j])*(B11[j,k]+B22[j,k]) 
    #       + (A22[i,j])*(B21[j,k]-B11[j,k]) 
    #       - (A11[i,j]+A12[i,j])*(B22[j,k]) 
    #       + (A12[i,j]-A22[i,j])*(B21[j,k]+B22[j,k]) )
    
    # This is getting complicated to expand manually.
    # Alternative: Just generate the terms for each M_k, but keep track of the final combination coefficients.
    # Actually, the standard way to implement this is to just expand the full polynomial.
    
    # Let's do it by expanding each of the 7 products into 27 terms, 
    # and then applying the final combination coefficients to the W matrix.
    
    # However, U and V also depend on the linear combinations.
    # So for M1, U has entries for A11 and A22. V has entries for B11 and B22.
    # The term is (A11[i,j] + A22[i,j]) * (B11[j,k] + B22[j,k]).
    # This expands to 4 scalar products:
    # 1. A11[i,j] * B11[j,k]
    # 2. A11[i,j] * B22[j,k]
    # 3. A22[i,j] * B11[j,k]
    # 4. A22[i,j] * B22[j,k]
    
    # Each of these is a rank-1 term in the global tensor.
    # We add these terms to U, V, W.
    # The W entry for this term will be multiplied by the coefficient from the final C combination.
    
    # Let's define the final combination coefficients for each M_k in each C_block.
    # C11: M1(+1), M4(+1), M5(-1), M7(+1)
    # C12: M3(+1), M5(+1)
    # C21: M2(+1), M4(+1)
    # C22: M1(+1), M2(-1), M3(+1), M6(+1)
    
    # We will iterate through each M_k, and for each scalar product within it,
    # we determine which C_block it contributes to, and with what sign.
    # Wait, M_k is a 3x3 matrix. It contributes to specific C_blocks.
    # M1 contributes to C11 (coeff 1) and C22 (coeff 1).
    # M2 contributes to C21 (coeff 1) and C22 (coeff -1).
    # M3 contributes to C12 (coeff 1) and C22 (coeff 1).
    # M4 contributes to C11 (coeff 1) and C21 (coeff 1).
    # M5 contributes to C11 (coeff -1) and C12 (coeff 1).
    # M6 contributes to C22 (coeff 1).
    # M7 contributes to C11 (coeff 1).
    
    # So for a term in M_k, we add it to U, V.
    # And we add it to W with the appropriate coefficient for each C_block it affects.
    # But W is a single matrix. A term in the decomposition corresponds to a specific (i, k) in the final C.
    # If M1 contributes to both C11 and C22, then the same scalar product in M1 generates terms in W for both C11 and C22?
    # No. The decomposition is of the full 6x6x6 tensor.
    # The tensor entry T[i, j, k] is the coefficient of A[i, j] B[j, k] in C[i, k].
    # If we use a rank-1 term U[:, c] V[:, c] W[:, c], it adds U[i, j] V[j, k] W[k, i] to T[i, j, k].
    # Note the W index convention: W[k*n+i, r].
    # So W[row, col] corresponds to C[col, row]?
    # Task says: W[k*n+i, r].
    # Standard tensor T_{ijk} = sum_r U_{ij, r} V_{jk, r} W_{ki, r}.
    # Here W index is k*n + i.
    # So for a fixed rank column r, the contribution to C[i, k] is U[i, j] V[j, k] W[k, i].
    
    # Let's refine the mapping.
    # For a scalar product A[a, b] * B[b, c] contributing to C[i, k]:
    # We need U[a, b, r] = coeff_A, V[b, c, r] = coeff_B, W[c, i, r] = coeff_C.
    # Wait, the indices in U, V, W are flattened.
    # U row: a*6 + b.
    # V row: b*6 + c.
    # W row: c*6 + i.
    
    # Let's implement this by generating all terms.
    
    # We'll use a list of tuples (u_idx, v_idx, w_idx, u_val, v_val, w_val)
    # and then aggregate them? No, rank is the number of columns.
    # If we just add terms, we might have multiple non-zeros in the same column?
    # No, each rank-1 term is a separate column.
    # So we just append to U, V, W lists.
    
    # But we pre-allocated U, V, W with fixed rank 189.
    # We need to be careful not to exceed rank 189.
    # Each M_k has 3x3x3 = 27 scalar products.
    # 7 * 27 = 189 terms.
    # So we will have exactly 189 columns.
    
    # Let's iterate.
    
    # Mapping of M_k to C_blocks and coefficients
    # M_k is index 0..6
    # C_block is (br, bc) in {0,1}x{0,1}
    # Coeff is the multiplier
    
    m_to_c = {
        0: { (0,0): 1, (1,1): 1 },   # M1
        1: { (1,0): 1, (1,1): -1 },  # M2
        2: { (0,1): 1, (1,1): 1 },   # M3
        3: { (0,0): 1, (1,0): 1 },   # M4
        4: { (0,0): -1, (0,1): 1 },  # M5
        5: { (1,1): 1 },             # M6
        6: { (0,0): 1 }              # M7
    }
    
    # A_blocks for M_k: list of (block_id, coeff)
    # B_blocks for M_k: list of (block_id, coeff)
    
    # Block ID to (row_offset, col_offset) for A and B
    # A11: (0,0), A12: (0,1), A21: (1,0), A22: (1,1)
    # B11: (0,0), B12: (0,1), B21: (1,0), B22: (1,1)
    
    block_offsets = {
        0: (0, 0), # 11
        1: (0, 1), # 12
        2: (1, 0), # 21
        3: (1, 1)  # 22
    }
    
    for m_idx, (a_comb, b_comb) in enumerate(products):
        # a_comb: list of (block_id, coeff)
        # b_comb: list of (block_id, coeff)
        
        # Expand a_comb and b_comb into scalar terms
        # For each (a_block, a_coeff) in a_comb
        # For each (b_block, b_coeff) in b_comb
        # We have a product of two 3x3 blocks
        # This product contributes to 27 scalar terms
        # For i in 0..2, j in 0..2, k in 0..2
        # Term: A[a_block][i, j] * B[b_block][j, k]
        
        for a_block_id, a_coeff in a_comb:
            a_row_off, a_col_off = block_offsets[a_block_id]
            for b_block_id, b_coeff in b_comb:
                b_row_off, b_col_off = block_offsets[b_block_id]
                
                # The product of these two blocks is a 3x3 matrix
                # Its (i, k) entry is sum_j A[a_block][i, j] B[b_block][j, k]
                # We decompose this into 27 rank-1 terms
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            # Global indices for U (A)
                            # A row: a_row_off*3 + i
                            # A col: a_col_off*3 + j
                            u_row = (a_row_off * 3 + i) * 6 + (a_col_off * 3 + j)
                            
                            # Global indices for V (B)
                            # B row: b_row_off*3 + j
                            # B col: b_col_off*3 + k
                            v_row = (b_row_off * 3 + j) * 6 + (b_col_off * 3 + k)
                            
                            # This term contributes to C blocks specified in m_to_c[m_idx]
                            # For each C_block (c_br, c_bc) with coeff c_coeff:
                            # The contribution is to C[c_br*3 + i, c_bc*3 + k]
                            # W index: C row = c_bc*3 + k, C col = c_br*3 + i
                            # W row: (C row)*6 + (C col) = (c_bc*3 + k)*6 + (c_br*3 + i)
                            
                            # Wait, we have one column in U, V, W per rank-1 term.
                            # But this term (A[i,j] B[j,k]) contributes to multiple C entries?
                            # No. In the tensor decomposition, a rank-1 term is defined by fixed indices.
                            # If the same A[i,j] and B[j,k] appear in multiple products that contribute to different C entries,
                            # we can't combine them into one rank-1 term unless the W vector is a sum.
                            # But W is a vector.
                            # The term is U[:, c] V[:, c] W[:, c].
                            # If U and V are fixed, W can have multiple non-zeros.
                            # So we can add multiple W entries to the same column c.
                            
                            # However, we are generating one column per scalar product in the expansion.
                            # If A[i,j] B[j,k] appears in M1 and M4, and both contribute to C11,
                            # we should combine them into one column.
                            # But M1 and M4 are different products.
                            # The expansion is linear.
                            # T = sum_terms coeff * A * B.
                            # We can group terms with same A, B indices.
                            
                            # Let's count how many times each (A_idx, B_idx) pair appears.
                            # Actually, it's easier to just add a new column for each scalar product in the expansion.
                            # This might lead to a higher rank if we don't combine.
                            # But the block-Strassen rank 189 assumes we DO combine?
                            # No, the rank of the tensor is the minimum number of rank-1 terms.
                            # If we generate 189 terms, and they are linearly independent, rank is 189.
                            # If some are dependent, rank is lower.
                            # But we want to achieve rank 189.
                            # So we just generate 189 columns.
                            
                            # For each (i, j, k) in the 3x3x3 loop:
                            # We add a column.
                            # U[u_row][col] = a_coeff
                            # V[v_row][col] = b_coeff
                            # W entries:
                            for c_br, c_bc in m_to_c[m_idx]:
                                c_coeff = m_to_c[m_idx][(c_br, c_bc)]
                                # C row: c_bc*3 + k
                                # C col: c_br*3 + i
                                w_row = (c_bc * 3 + k) * 6 + (c_br * 3 + i)
                                W[w_row][col] += c_coeff # Note: += because multiple M_k might contribute to same W row?
                                # No, each column is unique. But within one column, W can have multiple non-zeros.
                                # So we just set W[w_row][col] to the sum of contributions.
                                # But we are iterating over m_idx.
                                # For a fixed m_idx, i, j, k, we might have multiple c_blocks.
                                # We add them to the same column.
                            
                            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
