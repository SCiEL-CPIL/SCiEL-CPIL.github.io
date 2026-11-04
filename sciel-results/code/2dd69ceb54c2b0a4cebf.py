import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition
    # Based on Strassen (rank 7) for 2x2 and Schoolbook (rank 6) for 1x2x3
    # Total rank = 7 * 6 = 42
    
    rank = 42
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We define the terms (coeff, a_block, b_block) for each Strassen M_s
    # a_block is (row, col) in 1..2, 1..2
    # b_block is (row, col) in 1..2, 1..2
    
    strassen_terms = {
        1: [ (1, (1,1), (1,1)), (1, (1,1), (2,2)), (1, (2,2), (1,1)), (1, (2,2), (2,2)) ],
        2: [ (1, (2,1), (1,1)), (1, (2,2), (1,1)) ],
        3: [ (1, (1,1), (1,2)), (-1, (1,1), (2,2)) ],
        4: [ (1, (2,2), (2,1)), (-1, (2,2), (1,1)) ],
        5: [ (1, (1,1), (2,2)), (1, (1,2), (2,2)) ],
        6: [ (1, (2,1), (1,1)), (1, (2,1), (1,2)), (-1, (1,1), (1,1)), (-1, (1,1), (1,2)) ],
        7: [ (1, (1,2), (2,1)), (1, (1,2), (2,2)), (-1, (2,2), (2,1)), (-1, (2,2), (2,2)) ]
    }
    
    # Schoolbook terms for 1x2x3
    # A is 1x2, B is 2x3
    # U_t is 2x1, V_t is 6x1, W_t is 3x1
    # t=1..6
    # j_local in 1..2, k_local in 1..3
    # U[j_local, t] = 1 if j_local == t//3 + 1 ?
    # Let's just hardcode the 6 terms for 1x2x3
    # Term 1: A[0]B[0,0]
    # Term 2: A[0]B[0,1]
    # Term 3: A[0]B[0,2]
    # Term 4: A[1]B[1,0]
    # Term 5: A[1]B[1,1]
    # Term 6: A[1]B[1,2]
    
    # U shape (2, 6), V shape (6, 6), W shape (3, 6)
    # U[j, t], V[k, t], W[k, t]
    # Note: V index is j*B_cols + k? No, V is for B.
    # B is 2x3. Index (r, c) -> r*3+c.
    # U is for A. Index (c) -> c.
    # W is for C. Index (c) -> c.
    
    sb_U = np.zeros((2, 6))
    sb_V = np.zeros((6, 6))
    sb_W = np.zeros((3, 6))
    
    for t in range(6):
        j_loc = t // 3 # 0 or 1
        k_loc = t % 3   # 0, 1, 2
        
        sb_U[j_loc, t] = 1
        sb_V[j_loc * 3 + k_loc, t] = 1
        sb_W[k_loc, t] = 1
        
    col = 0
    for s in range(1, 8):
        for coeff, (a_r, a_c), (b_r, b_c) in strassen_terms[s]:
            for t in range(6):
                # Add contribution
                # A block (a_r, a_c) is at row a_r, block col a_c
                # Local col j_loc in 0..1
                # Global j = (a_c-1)*2 + j_loc
                # i = a_r
                
                # B block (b_r, b_c) is at row b_r, block col b_c
                # Local row j_loc_b (must match j_loc from A? No, A and B are different blocks)
                # Wait, in M_{1,2,3}, the A block is 1x2 and B block is 2x3.
                # The A block is A_{a_r, a_c}. The B block is B_{b_r, b_c}.
                # The product is A_{a_r, a_c} * B_{b_r, b_c}.
                # This requires the column index of A to match the row index of B.
                # A_{a_r, a_c} has columns 1,2.
                # B_{b_r, b_c} has rows 1,2.
                # So the "local" indices match.
                
                # For each t in schoolbook:
                j_loc_a = int(sb_U[:, t].nonzero()[0][0]) # 0 or 1
                # sb_U has 1 at j_loc_a
                # sb_V has 1 at j_loc_b * 3 + k_loc
                # We need to map j_loc_b to the row of B block.
                # The row of B block is b_r. The local row is j_loc_b.
                # Global row of B is (b_r-1)*2 + j_loc_b.
                
                # U index: i = a_r, j = (a_c-1)*2 + j_loc_a
                u_idx = (a_r - 1) * m + ((a_c - 1) * 2 + j_loc_a)
                U[u_idx][col] += coeff * sb_U[j_loc_a, t]
                
                # V index: j = (b_r-1)*2 + j_loc_b, k = (b_c-1)*3 + k_loc
                # sb_V index is j_loc_b * 3 + k_loc
                v_idx_flat = int(sb_V[:, t].nonzero()[0][0])
                j_loc_b = v_idx_flat // 3
                k_loc = v_idx_flat % 3
                v_idx = ((b_r - 1) * 2 + j_loc_b) * p + ((b_c - 1) * 3 + k_loc)
                V[v_idx][col] += coeff * sb_V[v_idx_flat, t]
                
                # W index: k = (b_c-1)*3 + k_loc, i = a_r
                # Trace convention W[k*n + i, r]
                w_idx_flat = int(sb_W[:, t].nonzero()[0][0])
                k_loc_w = w_idx_flat
                w_idx = ((b_c - 1) * 3 + k_loc_w) * n + (a_r - 1)
                W[w_idx][col] += coeff * sb_W[w_idx_flat, t]
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
