"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition: 7 blocks of 2x2, each using rank-7 Strassen
    # Total rank = 7 * 7 = 49
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    num_blocks = p // 2  # 7 blocks
    strassen_rank = 7
    rank = num_blocks * strassen_rank  # 49
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 49)
    V = [[0] * rank for _ in range(m * p)]  # (28, 49)
    W = [[0] * rank for _ in range(p * n)]  # (28, 49)
    
    # Strassen term definitions: (U_coeffs, V_coeffs, W_coeffs)
    # U index: i*m+j, V index: j*p+k, W index: k*n+i
    # For block b, B columns are 2b, 2b+1; C columns are 2b, 2b+1
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # Each entry: list of (index, coeff) for U, V, W
    # Term 0 (M1): (A00+A11)(B00+B11)
    # Term 1 (M2): (A10+A11)B00
    # Term 2 (M3): A00(B10-B11)
    # Term 3 (M4): A11(B01-B00)
    # Term 4 (M5): (A00+A01)B11
    # Term 5 (M6): (A10-A00)(B00+B01)
    # Term 6 (M7): (A01-A11)(B10+B11)
    
    strassen_terms = [
        # (U: [(idx, coeff), ...], V: [(col_offset, coeff), ...], W: [(col_offset, row_offset, coeff), ...])
        # For V: col is j*p + (2b + col_offset) where col_offset is 0 or 1
        # For W: col is (2b + col_offset)*n + row_offset
        # Term 0: M1
        ([(0, 1), (3, 1)], [(0, 1), (1, 1)], [(0, 0, 1), (1, 1, 1)]),
        # Term 1: M2
        ([(2, 1), (3, 1)], [(0, 1)], [(0, 1, 1)]),
        # Term 2: M3
        ([(0, 1)], [(0, 1), (1, -1)], [(1, 0, 1), (1, 1, 1)]),
        # Term 3: M4
        ([(3, 1)], [(1, 1), (0, -1)], [(0, 0, 1), (0, 1, 1)]),
        # Term 4: M5
        ([(0, 1), (1, 1)], [(1, 1)], [(0, 0, -1), (1, 0, 1)]),
        # Term 5: M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(1, 1, 1)]),
        # Term 6: M7
        ([(1, 1), (3, -1)], [(0, 1), (1, 1)], [(0, 0, 1), (1, 0, -1)]),
    ]
    
    for b in range(num_blocks):
        for t in range(strassen_rank):
            col = b * strassen_rank + t
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            
            # Set U
            for idx, coeff in u_coeffs:
                U[idx][col] = coeff
            
            # Set V: V[j*p + (2b + v_col_offset)]
            for v_col_offset, coeff in v_coeffs:
                # v_col_offset 0 means B[0, 2b], 1 means B[1, 2b+1] etc.
                # Actually need to map: the V index is j*p + k where k = 2b + offset
                # But my v_coeffs have (offset, coeff) where offset indicates which of the 2 columns
                # Let me re-examine: for block b, B's two columns are at global positions 2b and 2b+1
                # V index = j * p + k, where k is the global column index
                # So for B[0, 2b]: V[0 * 14 + 2b]
                # For B[1, 2b+1]: V[1 * 14 + 2b + 1]
                # My v_coeffs use (col_in_block, coeff) where col_in_block is 0 or 1
                # And I need to know which row j of B
                # Let me redefine: v_coeffs should be (j, col_in_block, coeff)
                pass
            
            # I need to redo the V and W definitions more carefully
            break
        break
    
    # Let me restart with clearer definitions
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms with explicit (U_idx, coeff), (V_row, V_col_offset, coeff), (W_col_offset, W_row, coeff)
    # V_row is j (0 or 1), V_col_offset is 0 or 1 (offset within the 2-column block)
    # W_col_offset is 0 or 1 (offset within the 2-column block), W_row is i (0 or 1)
    
    terms = [
        # (U_list, V_list, W_list)
        # Term 0: M1 = (A00+A11)(B00+B11)
        ([(0, 1), (3, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        # Term 1: M2 = (A10+A11)B00
        ([(2, 1), (3, 1)], [(0, 0, 1)], [(0, 1, 1)]),
        # Term 2: M3 = A00(B10-B11)
        ([(0, 1)], [(1, 0, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),
        # Term 3: M4 = A11(B01-B00)
        ([(3, 1)], [(0, 1, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)]),
        # Term 4: M5 = (A00+A01)B11
        ([(0, 1), (1, 1)], [(1, 1, 1)], [(0, 0, -1), (1, 0, 1)]),
        # Term 5: M6 = (A10-A00)(B00+B01)
        ([(2, 1), (0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        # Term 6: M7 = (A01-A11)(B10+B11)
        ([(1, 1), (3, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 0, -1)]),
    ]
    
    for b in range(num_blocks):
        for t in range(strassen_rank):
            col = b * strassen_rank + t
            u_list, v_list, w_list = terms[t]
            
            for u_idx, coeff in u_list:
                U[u_idx][col] = coeff
            
            for v_row, v_col_off, coeff in v_list:
                v_global_col = 2 * b + v_col_off
                V[v_row * p + v_global_col][col] = coeff
            
            for w_col_off, w_row, coeff in w_list:
                w_global_col = 2 * b + w_col_off
                W[w_global_col * n + w_row][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
