"""Rewritten program for Matrix Multiplication 2x5x6 using Strassen for 2x2 blocks."""
DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank estimate: 3 * 7 (Strassen for 2x2x2) + 2 * 18 (Schoolbook for 2x3x3) = 57
    rank = 57
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen helper for 2x2x2
    def add_strassen(i_offset, j_offset, k_offset):
        nonlocal col
        # i_offset: 0 for A1 (rows 0-1 of A)
        # j_offset: 0 for B1 (cols 0-1, 2-3, 4-5) -- Wait, j is inner dim.
        # A is 2x5. A1 is cols 0,1. A2 is cols 2,3,4.
        # B is 5x6. B1 is rows 0,1. B2 is rows 2,3,4.
        
        # Local indices i in 0..1, j in 0..1, k in 0..1
        # Global indices:
        # i_g = i + i_offset (i_offset is 0 for A1)
        # j_g = j + j_offset (j_offset is 0 for A1 cols 0,1)
        # k_g = k + k_offset (k_offset for B1 cols)
        
        # Strassen Products P1..P7
        # P1: (a00+a11)(b00+b11)
        # U: i=0, j=0; i=1, j=1
        # V: j=0, k=0; j=1, k=1
        # W: k=0, i=0; k=1, i=1? No, W is for C_ik.
        # C_00 gets P1. C_11 gets P1.
        
        # We define a helper to add a term
        def add_term(u_indices, v_indices, w_indices, w_val):
            nonlocal col
            for ui, uj in u_indices:
                U[i_offset * m + (uj + j_offset)][col] += ui
            for vj, vk in v_indices:
                V[(vj + j_offset) * p + (vk + k_offset)][col] += vj # Wait, V index is j*p+k. Value is coefficient.
            # Correction: V index is (j_global * p + k_global). The value is the coefficient in the linear form.
            # My add_term takes list of (index, coeff) or just indices if coeff is 1?
            # Let's pass lists of (global_index, coeff)
            
        # Let's restructure add_strassen to directly manipulate U, V, W with global indices
        
        # P1: (A00+A11)(B00+B11)
        # U coeffs: A00:1, A11:1. Global: (0, j_offset+0), (1, j_offset+1)
        # V coeffs: B00:1, B11:1. Global: (j_offset+0, k_offset+0), (j_offset+1, k_offset+1)
        # W coeffs: C00:1, C11:1. Global: (k_offset+0, 0), (k_offset+1, 1)
        
        # U updates
        U[i_offset * m + (0 + j_offset)][col] += 1
        U[i_offset * m + (1 + j_offset)][col] += 1
        # V updates
        V[(0 + j_offset) * p + (0 + k_offset)][col] += 1
        V[(1 + j_offset) * p + (1 + k_offset)][col] += 1
        # W updates
        W[(0 + k_offset) * n + i_offset][col] += 1 # C_00
        W[(1 + k_offset) * n + (i_offset + 1)][col] += 1 # C_11
        
        col += 1
        
        # P2: (A10+A11)B00
        # U: A10:1, A11:1
        # V: B00:1
        # W: C10:1
        U[i_offset * m + (0 + j_offset)][col] += 0
        U[i_offset * m + (1 + j_offset)][col] += 1
        U[i_offset * m + (1 + j_offset)][col] += 1 # A11 is row 1, col 1? No, A11 is A[1,1].
        # A10 is A[1,0]. A11 is A[1,1].
        # Local j for A10 is 0. Local j for A11 is 1.
        # So U[1, 0] += 1, U[1, 1] += 1.
        U[i_offset * m + (0 + j_offset)][col] = 0 # Reset if needed, but we are adding to fresh col
        U[i_offset * m + (1 + j_offset)][col] += 1
        U[i_offset * m + (1 + j_offset)][col] += 1
        # Wait, U index is i*m + j.
        # A10: i=1, j=0. U[1*m + 0].
        # A11: i=1, j=1. U[1*m + 1].
        # So:
        U[i_offset * m + (0 + j_offset)][col] = 0
        U[i_offset * m + (1 + j_offset)][col] = 1
        # This is getting messy with inline additions. Let's use a cleaner inner function.

    # Let's restart the construction logic cleanly.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_rank1(u_coeffs, v_coeffs, w_coeffs):
        """
        u_coeffs: dict {global_U_index: coeff}
        v_coeffs: dict {global_V_index: coeff}
        w_coeffs: dict {global_W_index: coeff}
        """
        for idx, c in u_coeffs.items():
            U[idx][col] += c
        for idx, c in v_coeffs.items():
            V[idx][col] += c
        for idx, c in w_coeffs.items():
            W[idx][col] += c
    
    def get_strassen_terms(i_off, j_off, k_off):
        """
        Generates the 7 rank-1 terms for 2x2 Strassen.
        i_off: offset for i in A (0 for A1)
        j_off: offset for j in A (0 for A1 cols 0,1)
        k_off: offset for k in B (0 for B1 cols 0,1)
        """
        terms = []
        
        # P1: (A00+A11)(B00+B11) -> C00 +1, C11 +1
        u = {(i_off)*m + (0+j_off): 1, (i_off+1)*m + (1+j_off): 1}
        v = {(0+j_off)*p + (0+k_off): 1, (1+j_off)*p + (1+k_off): 1}
        w = {(0+k_off)*n + i_off: 1, (1+k_off)*n + (i_off+1): 1}
        terms.append((u, v, w))
        
        # P2: (A10+A11)B00 -> C10 +1
        u = {(i_off+1)*m + (0+j_off): 1, (i_off+1)*m + (1+j_off): 1}
        v = {(0+j_off)*p + (0+k_off): 1}
        w = {(0+k_off)*n + (i_off+1): 1}
        terms.append((u, v, w))
        
        # P3: A00(B01-B11) -> C01 +1
        u = {(i_off)*m + (0+j_off): 1}
        v = {(0+j_off)*p + (1+k_off): 1, (1+j_off)*p + (1+k_off): -1}
        w = {(1+k_off)*n + i_off: 1}
        terms.append((u, v, w))
        
        # P4: A11(B10-B00) -> C00 +1, C10 +1
        u = {(i_off+1)*m + (1+j_off): 1}
        v = {(1+j_off)*p + (0+k_off): 1, (0+j_off)*p + (0+k_off): -1}
        w = {(0+k_off)*n + i_off: 1, (0+k_off)*n + (i_off+1): 1}
        terms.append((u, v, w))
        
        # P5: (A00+A01)B11 -> C00 -1, C01 +1
        u = {(i_off)*m + (0+j_off): 1, (i_off)*m + (1+j_off): 1}
        v = {(1+j_off)*p + (1+k_off): 1}
        w = {(1+k_off)*n + i_off: -1, (1+k_off)*n + i_off: 1} # C00 gets -1, C01 gets +1?
        # C01 is i=0, k=1. C00 is i=0, k=0.
        # P5 is in C00 (-1) and C01 (+1).
        # W index for C00: (0+k_off)*n + i_off.
        # W index for C01: (1+k_off)*n + i_off.
        w = {(0+k_off)*n + i_off: -1, (1+k_off)*n + i_off: 1}
        terms.append((u, v, w))
        
        # P6: (A10-A00)(B00+B01) -> C11 +1
        u = {(i_off+1)*m + (0+j_off): 1, (i_off)*m + (0+j_off): -1}
        v = {(0+j_off)*p + (0+k_off): 1, (0+j_off)*p + (1+k_off): 1}
        w = {(1+k_off)*n + (i_off+1): 1}
        terms.append((u, v, w))
        
        # P7: (A01-A11)(B10+B11) -> C00 +1
        u = {(i_off)*m + (1+j_off): 1, (i_off+1)*m + (1+j_off): -1}
        v = {(1+j_off)*p + (0+k_off): 1, (1+j_off)*p + (1+k_off): 1}
        w = {(0+k_off)*n + i_off: 1}
        terms.append((u, v, w))
        
        return terms

    # Process 3 Strassen blocks for A1 (2x2) * B1 (2x6)
    # A1 is cols 0,1 of A. j_off = 0. i_off = 0.
    # B1 is rows 0,1 of B.
    # B1a: cols 0,1. k_off = 0.
    # B1b: cols 2,3. k_off = 2.
    # B1c: cols 4,5. k_off = 4.
    
    for k_off in [0, 2, 4]:
        terms = get_strassen_terms(0, 0, k_off)
        for u, v, w in terms:
            add_rank1(u, v, w)
            col += 1
            
    # Process 2 Schoolbook blocks for A2 (2x3) * B2 (3x6)
    # A2 is cols 2,3,4 of A. j_off = 2. i_off = 0.
    # B2 is rows 2,3,4 of B. j_b_off = 2.
    # B2a: cols 0,1,2. k_off = 0.
    # B2b: cols 3,4,5. k_off = 3.
    
    for k_off in [0, 3]:
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    # Global indices
                    i_g = i
                    j_g = j + 2 # A2 starts at col 2
                    k_g = k + k_off
                    # B row index is j_g (since B2 rows correspond to A2 cols? No.)
                    # B is 5x6. B2 is rows 2,3,4.
                    # The multiplication is A2 (2x3) * B2 (3x6).
                    # Sum over j' in 0..2.
                    # A2[i, j'] * B2[j', k].
                    # Global A index: i * 5 + (j' + 2).
                    # Global B index: (j' + 2) * 6 + (k + k_off).
                    # Global C index: (k + k_off) * 2 + i.
                    
                    u_idx = i_g * m + j_g
                    v_idx = j_g * p + k_g
                    w_idx = k_g * n + i_g
                    
                    add_rank1({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
                    col += 1
                    
    return {"U": U, "V": V, "W": W}

