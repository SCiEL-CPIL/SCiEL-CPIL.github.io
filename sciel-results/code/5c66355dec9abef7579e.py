"""Rewritten program for 2x4x4 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # We use Strassen's algorithm for 2x2x2 multiplication.
    # The 7 terms for Strassen are defined by their support and coefficients.
    # Indices for 2x2 matrix are: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Strassen terms: (U_coeffs, V_coeffs, W_coeffs)
    # Each coeff is a dict {local_index: value}
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    # The 2x4x4 problem is decomposed into four 2x2x2 sub-problems.
    # A is 2x4, B is 4x4, C is 2x4.
    # C = A * B
    # C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
    # C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4]
    
    # Block 0: A_L (cols 0,1) * B_LL (rows 0,1, cols 0,1) -> C_L (cols 0,1)
    # Block 1: A_R (cols 2,3) * B_RL (rows 2,3, cols 0,1) -> C_L (cols 0,1)
    # Block 2: A_L (cols 0,1) * B_LR (rows 0,1, cols 2,3) -> C_R (cols 2,3)
    # Block 3: A_R (cols 2,3) * B_RR (rows 2,3, cols 2,3) -> C_R (cols 2,3)
    
    # For each block, we map local indices to global indices.
    # Global A index: i * 4 + j
    # Global B index: j * 4 + r
    # Global C index: r * 2 + i (Trace convention W[k*n+i, r] usually means C[r, i], 
    # but the prompt says W[k*n+i, r] where k=p, n=n. So W is p*n. 
    # Standard matrix mult C = A B. C[r, i] = sum A[i, k] B[k, r].
    # Tensor T_{i, k, r} = A[i, k] B[k, r] C[r, i].
    # The factor matrices U, V, W have shapes (n*m, rank), (m*p, rank), (p*n, rank).
    # U[i, k, t], V[k, r, t], W[r, i, t].
    # So W index is r * n + i. Here n=2. So W index is r * 2 + i.
    # This matches the prompt's "W[k*n+i,r]" if we interpret the first index of W as the flattened (r,i) pair?
    # Wait, prompt says "W[k*n+i,r]". This is confusing notation.
    # Usually W is (p*n, rank). The index into W is a single integer 0..p*n-1.
    # If the convention is C[r, i], then the index is r*n + i.
    # Let's stick to the previous program's logic which scored 1.1429 (valid).
    # Previous logic: g_idx_c = (c_col_offset + lj) * 2 + li.
    # Here c_col_offset is the column offset of C (which corresponds to 'r' in B's column).
    # lj is the local column index of C (0 or 1).
    # li is the local row index of C (0 or 1).
    # So global row of C = c_col_offset + lj.
    # global col of C = li.
    # Index = global_row * n + global_col = (c_col_offset + lj) * 2 + li.
    # This is consistent.
    
    blocks = [
        # (a_col_offset, b_row_offset, b_col_offset, c_col_offset, base_rank)
        (0, 0, 0, 0, 0),   # Block 0
        (2, 2, 0, 0, 7),   # Block 1
        (0, 0, 2, 2, 14),  # Block 2
        (2, 2, 2, 2, 21),  # Block 3
    ]

    for a_off, b_row_off, b_col_off, c_off, base_rank in blocks:
        for s in range(7):
            col = base_rank + s
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
            
            # Process U
            for local_idx, coeff in u_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx_a = li * 4 + (a_off + lj)
                U[g_idx_a][col] += coeff
                
            # Process V
            for local_idx, coeff in v_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx_b = (b_row_off + li) * 4 + (b_col_off + lj)
                V[g_idx_b][col] += coeff
                
            # Process W
            for local_idx, coeff in w_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx_c = (c_off + lj) * 2 + li
                W[g_idx_c][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
