import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    def add_strassen_block(col_start, c_a, r_b, c_b, r_c, c_c):
        # A indices (U)
        # A is 2x4. Index i*m + j.
        a_idx = 0 * m + c_a
        b_idx = 0 * m + (c_a + 1)
        c_idx = 1 * m + c_a
        d_idx = 1 * m + (c_a + 1)
        
        # B indices (V)
        # B is 4x4. Index j*p + k.
        e_idx = r_b * p + c_b
        f_idx = r_b * p + (c_b + 1)
        g_idx = (r_b + 1) * p + c_b
        h_idx = (r_b + 1) * p + (c_b + 1)
        
        # W indices (Output C)
        # C is 2x4 (n x p). Index k*n + i.
        # Z is 2x2.
        # Z[0,0] -> C[r_c, c_c]
        # Z[0,1] -> C[r_c, c_c+1]
        # Z[1,0] -> C[r_c+1, c_c]
        # Z[1,1] -> C[r_c+1, c_c+1]
        
        w_00 = c_c * n + r_c
        w_01 = (c_c + 1) * n + r_c
        w_10 = c_c * n + (r_c + 1)
        w_11 = (c_c + 1) * n + (r_c + 1)
        
        # Strassen Coefficients
        # M1: (a+d)(e+h)
        # M2: (c+d)e
        # M3: a(f-h)
        # M4: d(g-e)
        # M5: (a+b)h
        # M6: (c-a)(e+f)
        # M7: (b-d)(g+h)
        
        # Output
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # We define the 7 rank terms for this block
        # Each term r has U part, V part, and W part (coefficients for Z00..Z11)
        
        terms = []
        
        # M1
        terms.append({
            'U': [(a_idx, 1), (d_idx, 1)],
            'V': [(e_idx, 1), (h_idx, 1)],
            'W': [(w_00, 1), (w_11, 1)] # Z00, Z11
        })
        
        # M2
        terms.append({
            'U': [(c_idx, 1), (d_idx, 1)],
            'V': [(e_idx, 1)],
            'W': [(w_10, 1), (w_11, -1)] # Z10, Z11
        })
        
        # M3
        terms.append({
            'U': [(a_idx, 1)],
            'V': [(f_idx, 1), (h_idx, -1)],
            'W': [(w_01, 1), (w_11, 1)] # Z01, Z11
        })
        
        # M4
        terms.append({
            'U': [(d_idx, 1)],
            'V': [(g_idx, 1), (e_idx, -1)],
            'W': [(w_00, 1), (w_10, 1)] # Z00, Z10
        })
        
        # M5
        terms.append({
            'U': [(a_idx, 1), (b_idx, 1)],
            'V': [(h_idx, 1)],
            'W': [(w_00, -1), (w_01, 1)] # Z00, Z01
        })
        
        # M6
        terms.append({
            'U': [(c_idx, 1), (a_idx, -1)],
            'V': [(e_idx, 1), (f_idx, 1)],
            'W': [(w_11, 1)] # Z11
        })
        
        # M7
        terms.append({
            'U': [(b_idx, 1), (d_idx, -1)],
            'V': [(g_idx, 1), (h_idx, 1)],
            'W': [(w_00, 1)] # Z00
        })
        
        for t in terms:
            col = col_start
            for idx, val in t['U']:
                U[idx, col] += val
            for idx, val in t['V']:
                V[idx, col] += val
            for idx, val in t['W']:
                W[idx, col] += val
            col_start += 1
            
    # Block 1: A0 (cols 0-1), B00 (rows 0-1, cols 0-1) -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # Block 2: A1 (cols 2-3), B10 (rows 2-3, cols 0-1) -> C[0:2, 0:2]
    add_strassen_block(7, 2, 2, 0, 0, 0)
    
    # Block 3: A0 (cols 0-1), B01 (rows 0-1, cols 2-3) -> C[0:2, 2:4]
    add_strassen_block(14, 0, 0, 2, 0, 2)
    
    # Block 4: A1 (cols 2-3), B11 (rows 2-3, cols 2-3) -> C[0:2, 2:4]
    add_strassen_block(21, 2, 2, 2, 0, 2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
