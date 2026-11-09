import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # 4+1 Partition
    # Block 0: indices 0,1,2,3
    # Block 1: index 4
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        
        for idx, coeff in u_coeffs:
            U_col[idx] += coeff
        for idx, coeff in v_coeffs:
            V_col[idx] += coeff
        for idx, coeff in w_coeffs:
            W_col[idx] += coeff
            
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    def map_u(i, j):
        return i * m + j

    def map_v(j, k):
        return j * p + k

    def map_w(k, i):
        return k * n + i

    # Standard Strassen 2x2x2 terms
    # Returns list of (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (local_i, local_j, val)
    # v_coeffs: list of (local_j, local_k, val)
    # w_coeffs: list of (local_i, local_k, val)
    
    def strassen_2x2_terms(I_map, J_map, K_map):
        """
        Generates 7 terms for a 2x2x2 Strassen block.
        I_map, J_map, K_map: dict {local: global}
        """
        terms = []
        
        # Term 1: (a00+a11)(b00+b11)
        u = [(I_map[0], J_map[0], 1), (I_map[1], J_map[1], 1)]
        v = [(J_map[0], K_map[0], 1), (J_map[1], K_map[1], 1)]
        w = [(I_map[0], K_map[0], 1), (I_map[1], K_map[1], 1)]
        terms.append((u, v, w))
        
        # Term 2: (a10+a11)(b00)
        u = [(I_map[1], J_map[0], 1), (I_map[1], J_map[1], 1)]
        v = [(J_map[0], K_map[0], 1)]
        w = [(I_map[1], K_map[0], 1), (I_map[1], K_map[1], -1)]
        terms.append((u, v, w))
        
        # Term 3: (a00)(b01-b11)
        u = [(I_map[0], J_map[0], 1)]
        v = [(J_map[0], K_map[1], 1), (J_map[1], K_map[1], -1)]
        w = [(I_map[0], K_map[1], 1), (I_map[1], K_map[1], 1)]
        terms.append((u, v, w))
        
        # Term 4: (a11)(b10-b00)
        u = [(I_map[1], J_map[1], 1)]
        v = [(J_map[1], K_map[0], 1), (J_map[0], K_map[0], -1)]
        w = [(I_map[0], K_map[0], 1), (I_map[1], K_map[0], 1)]
        terms.append((u, v, w))
        
        # Term 5: (a00+a01)(b11)
        u = [(I_map[0], J_map[0], 1), (I_map[0], J_map[1], 1)]
        v = [(J_map[1], K_map[1], 1)]
        w = [(I_map[0], K_map[0], -1), (I_map[0], K_map[1], 1)]
        terms.append((u, v, w))
        
        # Term 6: (a10-a00)(b00+b01)
        u = [(I_map[1], J_map[0], 1), (I_map[0], J_map[0], -1)]
        v = [(J_map[0], K_map[0], 1), (J_map[0], K_map[1], 1)]
        w = [(I_map[1], K_map[1], 1)]
        terms.append((u, v, w))
        
        # Term 7: (a01-a11)(b10+b11)
        u = [(I_map[0], J_map[1], 1), (I_map[1], J_map[1], -1)]
        v = [(J_map[1], K_map[0], 1), (J_map[1], K_map[1], 1)]
        w = [(I_map[0], K_map[0], 1)]
        terms.append((u, v, w))
        
        return terms

    def add_strassen_4x4x4(offset_i, offset_j, offset_k):
        """
        Adds 49 terms for a 4x4x4 Strassen block.
        """
        # Sub-blocks: 00 (0,1), 01 (0,2), 10 (1,0), 11 (1,1) ? No.
        # Split 4 into 2+2.
        # I0: offset_i + [0, 1], I1: offset_i + [2, 3]
        # J0: offset_j + [0, 1], J1: offset_j + [2, 3]
        # K0: offset_k + [0, 1], K1: offset_k + [2, 3]
        
        I0 = [offset_i, offset_i + 1]
        I1 = [offset_i + 2, offset_i + 3]
        J0 = [offset_j, offset_j + 1]
        J1 = [offset_j + 2, offset_j + 3]
        K0 = [offset_k, offset_k + 1]
        K1 = [offset_k + 2, offset_k + 3]
        
        # Macro Strassen terms for 4x4
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Each Mk is a product of 2x2 matrices.
        # We decompose each Mk using Strassen 2x2.
        
        # Define the 2x2 "virtual" matrices for each Macro term.
        # For M1: A_virtual = [[A00, A11]], B_virtual = [[B00, B11]]? No.
        # M1 = (A00+A11)(B00+B11).
        # This is a sum of products.
        # We can view this as a single 2x2 matrix product if we define:
        # A_macro = [[A00, A11]], B_macro = [[B00, B11]]?
        # No, Strassen operates on blocks.
        
        # Let's use the explicit composition.
        # For each of the 7 Macro terms, we have coefficients for the 2x2 sub-blocks.
        # Then we apply the 7 Strassen 2x2 terms to the resulting 2x2 matrices.
        
        # Macro Term Definitions:
        # (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: {(si, sj): val} where si, sj in {0, 1}
        # B_coeffs: {(sj, sk): val}
        # C_coeffs: {(si, sk): val}
        
        macro_terms = [
            # M1: (A00+A11)(B00+B11)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2: (A10+A11)B00
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            # M3: A00(B01-B11)
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            # M4: A11(B10-B00)
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # M5: (A00+A01)B11
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
            # M6: (A10-A00)(B00+B01)
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # M7: (A01-A11)(B10+B11)
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        for A_mac, B_mac, C_mac in macro_terms:
            # Get the 7 Strassen 2x2 terms for the "virtual" 2x2 matrices
            # The virtual matrices are composed of the sub-blocks.
            # We need to map the local indices of the Strassen 2x2 terms to the global indices.
            
            # The Strassen 2x2 terms operate on local indices 0,1.
            # We will generate the 7 terms for a generic 2x2 Strassen block,
            # but we need to scale them by the macro coefficients.
            
            # Actually, it's easier to just iterate over the 7 Strassen 2x2 terms
            # and combine them with the macro coefficients.
            
            # Get standard Strassen 2x2 terms
            # We need a helper to get the terms for a specific mapping.
            # But here, the "local" indices of the Strassen 2x2 correspond to the sub-blocks.
            # So we can just use the standard Strassen 2x2 terms structure.
            
            # Let's define the standard Strassen 2x2 terms as lists of (u, v, w) entries
            # where u, v, w are dicts {(li, lj): val}
            
            S2 = [
                ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
                ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
                ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
                ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
                ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
                ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
                ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
            ]
            
            for u_s, v_s, w_s in S2:
                u_entries = []
                v_entries = []
                w_entries = []
                
                # Combine u_s with A_mac
                for (li, lj), u_val in u_s.items():
                    for (si, sj), a_val in A_mac.items():
                        val = u_val * a_val
                        if val != 0:
                            gi = I0[li] if si == 0 else I1[li]
                            gj = J0[lj] if sj == 0 else J1[lj]
                            u_entries.append((map_u(gi, gj), val))
                            
                # Combine v_s with B_mac
                for (lj, lk), v_val in v_s.items():
                    for (sj, sk), b_val in B_mac.items():
                        val = v_val * b_val
                        if val != 0:
                            gj = J0[lj] if sj == 0 else J1[lj]
                            gk = K0[lk] if sk == 0 else K1[lk]
                            v_entries.append((map_v(gj, gk), val))
                            
                # Combine w_s with C_mac
                for (li, lk), w_val in w_s.items():
                    for (si, sk), c_val in C_mac.items():
                        val = w_val * c_val
                        if val != 0:
                            gi = I0[li] if si == 0 else I1[li]
                            gk = K0[lk] if sk == 0 else K1[lk]
                            w_entries.append((map_w(gk, gi), val))
                            
                add_term(u_entries, v_entries, w_entries)

    def schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_term(
                        [(map_u(i, k), 1.0)],
                        [(map_v(k, j), 1.0)],
                        [(map_w(j, i), 1.0)]
                    )
                    
    # 1. Diagonal 4x4x4 block
    if time.time() - start_time > DEADLINE:
        pass
    else:
        add_strassen_4x4x4(0, 0, 0)
        
    # 2. Off-diagonal blocks
    # Indices:
    # Block 0: 0,1,2,3
    # Block 1: 4
    
    B0 = [0, 1, 2, 3]
    B1 = [4]
    
    # A01 * B10: i in B0, j in B1, k in B0
    # C01 += A01 B10
    # i in B0, j in B1, k in B0
    schoolbook_block(B0, B1, B0)
    
    # A00 * B01: i in B0, j in B1, k in B0? No.
    # C01 = A00 B01 + A01 B11
    # A00 B01: i in B0, j in B1, k in B0
    # Wait, k is the intermediate index.
    # C_ij = sum_k A_ik B_kj
    # C01: i in B0, j in B1.
    # k can be in B0 or B1.
    # If k in B0: A00 B01. i in B0, j in B1, k in B0.
    # If k in B1: A01 B11. i in B0, j in B1, k in B1.
    
    # So for C01 (i in B0, j in B1):
    # Term 1: k in B0. i in B0, j in B1, k in B0.
    # Term 2: k in B1. i in B0, j in B1, k in B1.
    
    # We already did A01 B10? No, that was for C00?
    # C00: i in B0, j in B0.
    # k in B0: A00 B00 (Done in Strassen).
    # k in B1: A01 B10. i in B0, j in B0, k in B1.
    
    # Let's list all off-diagonal contributions:
    
    # C00 (i in B0, j in B0):
    # k in B1: A01 B10. i in B0, j in B0, k in B1.
    schoolbook_block(B0, B0, B1)
    
    # C01 (i in B0, j in B1):
    # k in B0: A00 B01. i in B0, j in B1, k in B0.
    schoolbook_block(B0, B1, B0)
    # k in B1: A01 B11. i in B0, j in B1, k in B1.
    schoolbook_block(B0, B1, B1)
    
    # C10 (i in B1, j in B0):
    # k in B0: A10 B00. i in B1, j in B0, k in B0.
    schoolbook_block(B1, B0, B0)
    # k in B1: A11 B10. i in B1, j in B0, k in B1.
    schoolbook_block(B1, B0, B1)
    
    # C11 (i in B1, j in B1):
    # k in B0: A10 B01. i in B1, j in B1, k in B0.
    schoolbook_block(B1, B1, B0)
    # k in B1: A11 B11. i in B1, j in B1, k in B1.
    schoolbook_block(B1, B1, B1)
    
    rank = len(U_cols)
    
    # Convert to standard format
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
