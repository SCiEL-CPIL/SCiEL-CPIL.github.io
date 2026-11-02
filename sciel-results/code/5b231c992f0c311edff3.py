import time
import sys

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank) -> (16, rank)
    # V: (m*p, rank) -> (24, rank)
    # W: (p*n, rank) -> (24, rank)
    
    # We'll build terms dynamically
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        U_rows.append(u_coeffs)
        V_rows.append(v_coeffs)
        W_rows.append(w_coeffs)
    
    # Helper to convert sparse list to dense row
    def to_dense(coeffs, size):
        row = [0] * size
        for idx, c in coeffs:
            if 0 <= idx < size:
                row[idx] += c
        return row
    
    # --- Part 1: Strassen for 4x4x4 (B cols 0-3, C cols 0-3) ---
    # This gives 49 terms
    
    # Mapping helpers
    def a_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (x*2+r)*4 + (y*2+c)
        
    def b1_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (x*2+r)*6 + (y*2+c)
        
    def c1_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (y*2+c)*4 + (x*2+r)
    
    # Strassen 2x2x2 terms
    strassen_2x2_terms = [
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),  # S1
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)]),               # S2
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)]),              # S3
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)]),              # S4
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)]),               # S5
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),  # S6
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)])   # S7
    ]
    
    result_map = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0,1): [(2, 1), (4, 1)],
        (1,0): [(1, 1), (3, 1)],
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
    big_combos = [
        ([('00', 1), ('11', 1)], [('00', 1), ('11', 1)]),
        ([('10', 1), ('11', 1)], [('00', 1)]),
        ([('00', 1)], [('01', 1), ('11', -1)]),
        ([('11', 1)], [('10', 1), ('00', -1)]),
        ([('00', 1), ('01', 1)], [('11', 1)]),
        ([('10', 1), ('00', -1)], [('00', 1), ('01', 1)]),
        ([('01', 1), ('11', -1)], [('10', 1), ('11', 1)])
    ]
    
    big_result_map = {
        '00': [(0, 1), (3, 1), (4, -1), (6, 1)],
        '01': [(2, 1), (4, 1)],
        '10': [(1, 1), (3, 1)],
        '11': [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
    for big_idx, (a_parts, b_parts) in enumerate(big_combos):
        for s_idx in range(7):
            x_coeffs, y_coeffs = strassen_2x2_terms[s_idx]
            
            u_coeffs = []
            for (rx, cx), cx_c in x_coeffs:
                for a_block, a_sign in a_parts:
                    idx = a_idx(a_block, rx, cx)
                    u_coeffs.append((idx, a_sign * cx_c))
            
            v_coeffs = []
            for (ry, cy), cy_c in y_coeffs:
                for b_block, b_sign in b_parts:
                    idx = b1_idx(b_block, ry, cy)
                    v_coeffs.append((idx, b_sign * cy_c))
            
            w_coeffs = []
            for c_block, big_contribs in big_result_map.items():
                big_coeff = 0
                for m_idx, m_c in big_contribs:
                    if m_idx == big_idx:
                        big_coeff = m_c
                        break
                if big_coeff == 0:
                    continue
                
                for rc, s_contribs in result_map.items():
                    small_coeff = 0
                    for s_check, s_c in s_contribs:
                        if s_check == s_idx:
                            small_coeff = s_c
                            break
                    if small_coeff == 0:
                        continue
                    
                    idx = c1_idx(c_block, rc[0], rc[1])
                    w_coeffs.append((idx, big_coeff * small_coeff))
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
    
    # --- Part 2: Optimized 4x4x2 (B cols 4-5, C cols 4-5) ---
    # Split A into 2x2 blocks, B2 (4x2) into two 2x2 blocks
    # C_top = A00*B20 + A01*B21 (each 2x2x2, 7 terms each = 14)
    # C_bottom = A10*B20 + A11*B21 (each 2x2x2, 7 terms each = 14)
    # Total: 28 terms
    
    # B20: j in [0,1], k in [4,5] -> global j*6 + (4+k)
    # B21: j in [2,3], k in [4,5] -> global j*6 + (4+k)
    
    # For C_top: i in [0,1], k in [4,5] -> global (4+k)*4 + i
    # For C_bottom: i in [2,3], k in [4,5] -> global (4+k)*4 + i
    
    # A00: i in [0,1], j in [0,1]
    # A01: i in [0,1], j in [2,3]
    # A10: i in [2,3], j in [0,1]
    # A11: i in [2,3], j in [2,3]
    
    # Each 2x2x2 product: X (2x2) * Y (2x2) = Z (2x2)
    # Using Strassen with 7 terms
    
    # For C_top = A00*B20 + A01*B21:
    # This is sum of two 2x2x2 products. We compute each separately.
    
    def add_2x2x2_product(a_block, b_cols, c_block, c_row_offset):
        """Add a 2x2x2 Strassen product.
        a_block: '00','01','10','11' for A
        b_cols: list of 4 (j,k) pairs for the 2x2 B block
        c_block: which C rows (0 for top, 1 for bottom)
        c_row_offset: row offset for C (0 or 2)
        """
        # b_cols: [(j0,k0), (j0,k1), (j1,k0), (j1,k1)]
        # For B20: j in [0,1], k in [4,5]
        # For B21: j in [2,3], k in [4,5]
        
        for s_idx in range(7):
            x_coeffs, y_coeffs = strassen_2x2_terms[s_idx]
            
            u_coeffs = []
            for (rx, cx), cx_c in x_coeffs:
                idx = a_idx(a_block, rx, cx)
                u_coeffs.append((idx, cx_c))
            
            v_coeffs = []
            for (ry, cy), cy_c in y_coeffs:
                j, k = b_cols[ry * 2 + cy]
                idx = j * 6 + k
                v_coeffs.append((idx, cy_c))
            
            w_coeffs = []
            for rc, s_contribs in result_map.items():
                small_coeff = 0
                for s_check, s_c in s_contribs:
                    if s_check == s_idx:
                        small_coeff = s_c
                        break
                if small_coeff == 0:
                    continue
                
                # C index: i = c_row_offset + rc[0], k = 4 + rc[1]
                i = c_row_offset + rc[0]
                k = 4 + rc[1]
                idx = k * 4 + i
                w_coeffs.append((idx, small_coeff))
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
    
    # C_top = A00*B20 + A01*B21
    b20_cols = [(0,4), (0,5), (1,4), (1,5)]
    b21_cols = [(2,4), (2,5), (3,4), (3,5)]
    
    add_2x2x2_product('00', b20_cols, 'top', 0)
    add_2x2x2_product('01', b21_cols, 'top', 0)
    add_2x2x2_product('10', b20_cols, 'bottom', 2)
    add_2x2x2_product('11', b21_cols, 'bottom', 2)
    
    # Convert sparse to dense
    rank = len(U_rows)
    U = [to_dense(u, n * m) for u in U_rows]
    V = [to_dense(v, m * p) for v in V_rows]
    W = [to_dense(w, p * n) for w in W_rows]
    
    return {"U": U, "V": V, "W": W}
