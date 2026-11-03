"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part for first 4 columns of B
    # Blocks are 2x2
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # Define Strassen terms
    # Each term is (A_coeffs, B_coeffs, C_coeffs)
    # Coeffs are dict: {block_idx: coeff}
    # block_idx: 00, 01, 10, 11
    
    strassen_terms = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col = 0
    for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
        # For each of the 4 schoolbook terms of the 2x2 product
        for a_local in range(2):
            for b_local in range(2):
                # This term corresponds to A[., .] * B[., .] -> C[., .]
                # The intermediate index is not fixed in the sum, but in rank-1 decomposition
                # of 2x2x2x2, we sum over c_local.
                # Wait, the rank-1 tensor is U_a,c * V_c,b * W_b,a.
                # So for a fixed a_local, b_local, we sum over c_local.
                # But in the decomposition, each (a, b, c) is a separate rank-1 term.
                # So we iterate c_local as well.
                for c_local in range(2):
                    # U component
                    u_val = 0
                    for (bi, bj), coeff in a_coeffs.items():
                        if True: # A block bi, bj
                            # Local index in A block: row a_local, col c_local
                            # Global row: 2*bi + a_local
                            # Global col: 2*bj + c_local
                            r = 2 * bi + a_local
                            c_idx = 2 * bj + c_local
                            u_val += coeff # Wait, U is just 1 if present?
                            # No, U is a vector. For a specific rank-1 term, U has non-zero
                            # entries where the A block is selected.
                            # But the A block itself is a 2x2 matrix.
                            # The term is A[2bi+a, 2bj+c] * B[2bi'+c', 2bj'+b] ...
                            # Actually, the Strassen formula is linear.
                            # A_k = sum coeff * A_block.
                            # The product A_k B_k = sum coeff_a coeff_b A_block_a B_block_b.
                            # Each A_block_a B_block_b is a 2x2 product.
                            # So we can just add the contributions.
                            pass
                            
                    # Let's re-structure.
                    # For each Strassen term, we have 4 sub-products (from schoolbook 2x2).
                    # Each sub-product corresponds to a specific (a_local, b_local) pair in the result?
                    # No, schoolbook 2x2 has 4 terms, each corresponding to a specific (a, b, c) triple.
                    # Term (a, b, c): U[a, c]=1, V[c, b]=1, W[b, a]=1.
                    # In our context:
                    # U has non-zeros in A blocks specified by a_coeffs.
                    # V has non-zeros in B blocks specified by b_coeffs.
                    # W has non-zeros in C blocks specified by c_coeffs.
                    
                    # For a specific schoolbook index (a_local, b_local, c_local):
                    # U vector:
                    #   For each (bi, bj) in a_coeffs:
                    #      Set U[global_row, global_col] += coeff_a
                    #      global_row = 2*bi + a_local
                    #      global_col = 2*bj + c_local
                    # V vector:
                    #   For each (bi, bj) in b_coeffs:
                    #      Set V[global_row, global_col] += coeff_b
                    #      global_row = 2*bi + c_local
                    #      global_col = 2*bj + b_local
                    # W vector:
                    #   For each (bi, bj) in c_coeffs:
                    #      Set W[global_row, global_col] += coeff_c
                    #      global_row = 2*bi + b_local  (This is 'k' in W[k*n+i]?)
                    #      global_col = 2*bj + a_local  (This is 'i' in W[k*n+i]?)
                    #      Wait, W index is k*n + i.
                    #      C block bi, bj has rows 2bi..2bi+1, cols 2bj..2bj+1.
                    #      Local row b_local, col a_local.
                    #      Global row (k) = 2*bi + b_local
                    #      Global col (i) = 2*bj + a_local
                    #      W index = (2*bi + b_local) * 4 + (2*bj + a_local)
                    
                    u_vec = [0] * rank
                    v_vec = [0] * rank
                    w_vec = [0] * rank
                    
                    for (bi, bj), coeff_a in a_coeffs.items():
                        r = 2 * bi + a_local
                        c = 2 * bj + c_local
                        idx = r * 4 + c
                        u_vec[idx] += coeff_a
                        
                    for (bi, bj), coeff_b in b_coeffs.items():
                        r = 2 * bi + c_local
                        c = 2 * bj + b_local
                        idx = r * 5 + c
                        v_vec[idx] += coeff_b
                        
                    for (bi, bj), coeff_c in c_coeffs.items():
                        r = 2 * bi + b_local # k
                        c = 2 * bj + a_local # i
                        idx = r * 4 + c
                        w_vec[idx] += coeff_c
                        
                    for r_idx in range(rank):
                        U[r_idx][col] = u_vec[r_idx]
                        V[r_idx][col] = v_vec[r_idx]
                        W[r_idx][col] = w_vec[r_idx]
                        
                    col += 1
                    
    # Schoolbook part for last column (col 4 of B)
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
