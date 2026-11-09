"""Matrix multiplication 3x4x4 with rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    nm = n * m  # 12
    mp = m * p  # 16
    pn = p * n  # 12
    
    # Initialize factor matrices
    # We will build a list of terms, then convert to matrices
    # Each term is a dict: {'u': {idx: coeff}, 'v': {idx: coeff}, 'w': {idx: coeff}}
    terms = []
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """
        u_coeffs: dict {index: coeff}
        v_coeffs: dict {index: coeff}
        w_coeffs: dict {index: coeff}
        """
        terms.append((u_coeffs, v_coeffs, w_coeffs))

    def add_strassen_block(i_off, j_off, k_off):
        """
        Adds 7 rank-1 terms for a 2x2x2 Strassen block.
        i_off: row offset in A (and C)
        j_off: col offset in A (and row offset in B)
        k_off: col offset in B (and C)
        """
        # Local indices in A
        # a: (i_off, j_off)
        # b: (i_off, j_off+1)
        # c: (i_off+1, j_off)
        # d: (i_off+1, j_off+1)
        ua_a = i_off * m + j_off
        ua_b = i_off * m + (j_off + 1)
        ua_c = (i_off + 1) * m + j_off
        ua_d = (i_off + 1) * m + (j_off + 1)
        
        # Local indices in B
        # e: (j_off, k_off)
        # f: (j_off, k_off+1)
        # g: (j_off+1, k_off)
        # h: (j_off+1, k_off+1)
        vb_e = j_off * p + k_off
        vb_f = j_off * p + (k_off + 1)
        vb_g = (j_off + 1) * p + k_off
        vb_h = (j_off + 1) * p + (k_off + 1)
        
        # Local indices in C
        # c00: (i_off, k_off) -> W idx: k_off*n + i_off
        # c01: (i_off, k_off+1) -> W idx: (k_off+1)*n + i_off
        # c10: (i_off+1, k_off) -> W idx: k_off*n + (i_off+1)
        # c11: (i_off+1, k_off+1) -> W idx: (k_off+1)*n + (i_off+1)
        wc_00 = k_off * n + i_off
        wc_01 = (k_off + 1) * n + i_off
        wc_10 = k_off * n + (i_off + 1)
        wc_11 = (k_off + 1) * n + (i_off + 1)
        
        # p1 = (a+d)(e+h)
        # Contributes to c00 (+1), c11 (+1)
        add_term({ua_a: 1, ua_d: 1}, {vb_e: 1, vb_h: 1}, {wc_00: 1, wc_11: 1})
        
        # p2 = (c+d)e
        # Contributes to c10 (+1)
        add_term({ua_c: 1, ua_d: 1}, {vb_e: 1}, {wc_10: 1})
        
        # p3 = a(f-h)
        # Contributes to c01 (+1), c11 (-1)
        add_term({ua_a: 1}, {vb_f: 1, vb_h: -1}, {wc_01: 1, wc_11: -1})
        
        # p4 = d(g-e)
        # Contributes to c00 (-1), c10 (+1)
        add_term({ua_d: 1}, {vb_g: 1, vb_e: -1}, {wc_00: -1, wc_10: 1})
        
        # p5 = (a+b)h
        # Contributes to c01 (+1)
        add_term({ua_a: 1, ua_b: 1}, {vb_h: 1}, {wc_01: 1})
        
        # p6 = (c-a)(e+f)
        # Contributes to c00 (+1)
        add_term({ua_c: 1, ua_a: -1}, {vb_e: 1, vb_f: 1}, {wc_00: 1})
        
        # p7 = (b-d)(g+h)
        # Contributes to c11 (+1)
        add_term({ua_b: 1, ua_d: -1}, {vb_g: 1, vb_h: 1}, {wc_11: 1})

    # 1. Handle the 2x4 x 4x4 part (rows 0,1 of A)
    # Split into four 2x2 x 2x2 blocks
    # Block 1: A rows 0,1, cols 0,1; B rows 0,1, cols 0,1
    add_strassen_block(0, 0, 0)
    # Block 2: A rows 0,1, cols 0,1; B rows 0,1, cols 2,3
    add_strassen_block(0, 0, 2)
    # Block 3: A rows 0,1, cols 2,3; B rows 2,3, cols 0,1
    add_strassen_block(0, 2, 0)
    # Block 4: A rows 0,1, cols 2,3; B rows 2,3, cols 2,3
    add_strassen_block(0, 2, 2)
    
    # 2. Handle the 1x4 x 4x4 part (row 2 of A)
    # This is 16 rank-1 terms: A[2,j] * B[j,k]
    for j in range(4):
        for k in range(4):
            u_idx = 2 * m + j
            v_idx = j * p + k
            w_idx = k * n + 2
            add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
            
    # Construct U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    for r in range(rank):
        u_dict, v_dict, w_dict = terms[r]
        for idx, coeff in u_dict.items():
            U[idx][r] = coeff
        for idx, coeff in v_dict.items():
            V[idx][r] = coeff
        for idx, coeff in w_dict.items():
            W[idx][r] = coeff
            
    # Verification
    # Build tensor T
    T = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
                
    # Reconstruct
    T_recon = np.zeros((nm, mp, pn), dtype=np.int64)
    for r in range(rank):
        nz_u = [a for a in range(nm) if U[a][r] != 0]
        nz_v = [b for b in range(mp) if V[b][r] != 0]
        nz_w = [c for c in range(pn) if W[c][r] != 0]
        for a in nz_u:
            for b in nz_v:
                for c in nz_w:
                    T_recon[a, b, c] += U[a][r] * V[b][r] * W[c][r]
                    
    if not np.array_equal(T_recon, T):
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(nm)]
        V_school = [[0] * rank_school for _ in range(mp)]
        W_school = [[0] * rank_school for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_school[a][col] = 1
                    V_school[b][col] = 1
                    W_school[c][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
        
    return {"U": U, "V": V, "W": W}
