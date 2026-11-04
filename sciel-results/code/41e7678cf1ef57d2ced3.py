import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx):
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[u_idx] = 1
        v[v_idx] = 1
        w[w_idx] = 1
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    # Strassen 2x2x2 decomposition
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[p1, p2], [p3, p4]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # We need to map these to the global indices.
    # For a 2x2x2 block at (i_offset, j_offset, k_offset):
    # A indices: i_offset, i_offset+1 ; j_offset, j_offset+1
    # B indices: j_offset, j_offset+1 ; k_offset, k_offset+1
    # C indices: k_offset, k_offset+1 ; i_offset, i_offset+1
    
    def add_strassen_2x2x2(i_off, j_off, k_off):
        # A indices
        a_idx = i_off * m + j_off
        b_idx = i_off * m + (j_off + 1)
        c_idx = (i_off + 1) * m + j_off
        d_idx = (i_off + 1) * m + (j_off + 1)
        
        # B indices
        e_idx = j_off * p + k_off
        f_idx = j_off * p + (k_off + 1)
        g_idx = (j_off + 1) * p + k_off
        h_idx = (j_off + 1) * p + (k_off + 1)
        
        # C indices (W index: k*n + i)
        c11_idx = k_off * n + i_off
        c12_idx = (k_off + 1) * n + i_off
        c21_idx = k_off * n + (i_off + 1)
        c22_idx = (k_off + 1) * n + (i_off + 1)
        
        # m1 = (a+d)(e+h)
        # This is not a single rank-1 term in the standard basis, 
        # but Strassen combines them.
        # The standard Strassen algorithm uses 7 multiplications.
        # We need to represent the 7 multiplications as rank-1 terms in the tensor decomposition.
        # Each multiplication is of the form (linear combo of A) * (linear combo of B) -> linear combo of C.
        
        # Term 1: (a+d) * (e+h)
        # U: a+d -> indices a_idx, d_idx with coeff 1
        # V: e+h -> indices e_idx, h_idx with coeff 1
        # W: c11, c22 -> c11 gets +1, c22 gets -1 (from m1 - m2 + m3 + m6? No)
        # Let's trace the contributions.
        
        # m1 contributes to:
        # C11: +1
        # C22: +1
        # (Wait, C22 = m1 - m2 + m3 + m6. So m1 is +1 in C22)
        
        # Let's define the 7 terms explicitly.
        
        # T1: (a+d) * (e+h)
        u1 = {a_idx: 1, d_idx: 1}
        v1 = {e_idx: 1, h_idx: 1}
        w1 = {c11_idx: 1, c22_idx: 1}
        
        # T2: (c+d) * e
        u2 = {c_idx: 1, d_idx: 1}
        v2 = {e_idx: 1}
        w2 = {c21_idx: 1, c22_idx: -1} # C21 = m2+m4, C22 = ...-m2...
        
        # T3: a * (f-h)
        u3 = {a_idx: 1}
        v3 = {f_idx: 1, h_idx: -1}
        w3 = {c12_idx: 1, c22_idx: 1} # C12 = m3+m5, C22 = ...+m3...
        
        # T4: d * (g-e)
        u4 = {d_idx: 1}
        v4 = {g_idx: 1, e_idx: -1}
        w4 = {c11_idx: -1, c21_idx: 1} # C11 = m1+m4-m5+m7, C21 = m2+m4
        
        # T5: (a+b) * h
        u5 = {a_idx: 1, b_idx: 1}
        v5 = {h_idx: 1}
        w5 = {c11_idx: -1, c12_idx: 1} # C11 = ...-m5..., C12 = m3+m5
        
        # T6: (c-a) * (e+f)
        u6 = {c_idx: 1, a_idx: -1}
        v6 = {e_idx: 1, f_idx: 1}
        w6 = {c22_idx: 1} # C22 = ...+m6...
        
        # T7: (b-d) * (g+h)
        u7 = {b_idx: 1, d_idx: -1}
        v7 = {g_idx: 1, h_idx: 1}
        w7 = {c11_idx: 1} # C11 = ...+m7...
        
        # Add these 7 terms
        for u_map, v_map, w_map in [(u1, v1, w1), (u2, v2, w2), (u3, v3, w3), (u4, v4, w4), (u5, v5, w5), (u6, v6, w6), (u7, v7, w7)]:
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            for idx, val in u_map.items(): u[idx] = val
            for idx, val in v_map.items(): v[idx] = val
            for idx, val in w_map.items(): w[idx] = val
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)

    # Schoolbook 2x2x1 decomposition
    # A = [[a, b], [c, d]], B = [e, f]^T (2x1)
    # C = [a*e+b*f, c*e+d*f]
    # 4 terms.
    
    def add_schoolbook_2x2x1(i_off, j_off, k_off):
        a_idx = i_off * m + j_off
        b_idx = i_off * m + (j_off + 1)
        c_idx = (i_off + 1) * m + j_off
        d_idx = (i_off + 1) * m + (j_off + 1)
        
        e_idx = j_off * p + k_off
        f_idx = (j_off + 1) * p + k_off
        
        c1_idx = k_off * n + i_off
        c2_idx = k_off * n + (i_off + 1)
        
        # T1: a*e
        u1 = [0]*(n*m); v1 = [0]*(m*p); w1 = [0]*(p*n)
        u1[a_idx] = 1; v1[e_idx] = 1; w1[c1_idx] = 1
        U_cols.append(u1); V_cols.append(v1); W_cols.append(w1)
        
        # T2: b*f
        u2 = [0]*(n*m); v2 = [0]*(m*p); w2 = [0]*(p*n)
        u2[b_idx] = 1; v2[f_idx] = 1; w2[c1_idx] = 1
        U_cols.append(u2); V_cols.append(v2); W_cols.append(w2)
        
        # T3: c*e
        u3 = [0]*(n*m); v3 = [0]*(m*p); w3 = [0]*(p*n)
        u3[c_idx] = 1; v3[e_idx] = 1; w3[c2_idx] = 1
        U_cols.append(u3); V_cols.append(v3); W_cols.append(w3)
        
        # T4: d*f
        u4 = [0]*(n*m); v4 = [0]*(m*p); w4 = [0]*(p*n)
        u4[d_idx] = 1; v4[f_idx] = 1; w4[c2_idx] = 1
        U_cols.append(u4); V_cols.append(v4); W_cols.append(w4)

    # Part 1: Columns 1-4 (indices 0-3)
    # 4 blocks of 2x2x4
    # Block (0,0): i 0-1, j 0-1, k 0-3
    # Use 2 Strassen: one for k 0-1, one for k 2-3
    add_strassen_2x2x2(0, 0, 0)
    add_strassen_2x2x2(0, 0, 2)
    
    # Block (0,1): i 0-1, j 2-3, k 0-3
    add_strassen_2x2x2(0, 2, 0)
    add_strassen_2x2x2(0, 2, 2)
    
    # Block (1,0): i 2-3, j 0-1, k 0-3
    add_strassen_2x2x2(2, 0, 0)
    add_strassen_2x2x2(2, 0, 2)
    
    # Block (1,1): i 2-3, j 2-3, k 0-3
    add_strassen_2x2x2(2, 2, 0)
    add_strassen_2x2x2(2, 2, 2)
    
    # Part 2: Columns 5-7 (indices 4-6)
    # 4 blocks of 2x2x3
    # Use 1 Strassen (for k 4-5) + 1 Schoolbook (for k 6)
    
    # Block (0,0): i 0-1, j 0-1, k 4-6
    add_strassen_2x2x2(0, 0, 4)
    add_schoolbook_2x2x1(0, 0, 6)
    
    # Block (0,1): i 0-1, j 2-3, k 4-6
    add_strassen_2x2x2(0, 2, 4)
    add_schoolbook_2x2x1(0, 2, 6)
    
    # Block (1,0): i 2-3, j 0-1, k 4-6
    add_strassen_2x2x2(2, 0, 4)
    add_schoolbook_2x2x1(2, 0, 6)
    
    # Block (1,1): i 2-3, j 2-3, k 4-6
    add_strassen_2x2x2(2, 2, 4)
    add_schoolbook_2x2x1(2, 2, 6)
    
    rank = len(U_cols)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
