import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Total rank is 66
    # Block 1 (i=0): 2 * M_{1,2,6} -> 2 * 12 = 24
    # Block 2 (i=1,2): 2 * M_{2,2,6} -> 2 * 21 = 42
    total_rank = 66
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Helper to fill M_{1,2,6}
    # i_offset: global row index offset for A (0 for i=0)
    # j_offset: global col index offset for A (0 for j=0, 2 for j=2)
    # j_count: number of columns in A (2)
    # p_count: number of columns in B (6)
    # c_offset: start index for this block
    def fill_m126(i_offset, j_offset, c_offset):
        # A is 1 x 2. B is 2 x 6.
        # i is fixed to i_offset (which is 0).
        # j is j_offset, j_offset+1.
        # k is 0..5.
        c = c_offset
        for j_local in range(2):
            j_global = j_offset + j_local
            for k in range(6):
                # U[i_global * m + j_global, c] = 1
                U[i_offset * m + j_global, c] = 1.0
                # V[j_global * p + k, c] = 1
                V[j_global * p + k, c] = 1.0
                # W[k * n + i_global, c] = 1
                W[k * n + i_offset, c] = 1.0
                c += 1
                
    # Helper to fill M_{2,2,6}
    # i_offset: global row offset for A (1 for i=1,2)
    # j_offset: global col offset for A (0 for j=0,1 or 2 for j=2,3)
    # c_offset: start index for this block
    def fill_m226(i_offset, j_offset, c_offset):
        # A is 2 x 2. B is 2 x 6.
        # i is i_offset, i_offset+1.
        # j is j_offset, j_offset+1.
        # B is split into 3 blocks of 2x2.
        # b is 0, 1, 2.
        # k is 2b, 2b+1.
        
        # Strassen coefficients for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # A indices: local i' in {0,1}, local j' in {0,1}
        # Global i = i_offset + i'
        # Global j = j_offset + j'
        
        # B indices: local j' in {0,1}, local k' in {0,1}
        # Global j = j_offset + j' (Note: B rows correspond to A cols)
        # Global k = 2b + k'
        
        strassen_terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # U: dict (i', j') -> val
            # V: dict (j', k') -> val
            # W: dict (k', i') -> val
            {
                'U': {(0,0): 1, (1,1): 1},
                'V': {(0,0): 1, (1,1): 1},
                'W': {(0,0): 1, (1,1): 1}
            },
            {
                'U': {(1,0): 1, (1,1): 1},
                'V': {(0,0): 1},
                'W': {(0,1): 1} # C10 -> k=0, i=1 -> (0,1)
            },
            {
                'U': {(0,0): 1},
                'V': {(0,1): 1, (1,1): -1},
                'W': {(1,0): 1} # C01 -> k=1, i=0 -> (1,0)
            },
            {
                'U': {(1,1): 1},
                'V': {(1,0): 1, (0,0): -1},
                'W': {(1,1): 1} # C11 -> k=1, i=1 -> (1,1)
            },
            {
                'U': {(0,0): 1, (0,1): 1},
                'V': {(1,1): 1},
                'W': {(1,0): 1} # C01
            },
            {
                'U': {(1,0): 1, (0,0): -1},
                'V': {(0,0): 1, (0,1): 1},
                'W': {(0,1): 1} # C10
            },
            {
                'U': {(0,1): 1, (1,1): -1},
                'V': {(1,0): 1, (1,1): 1},
                'W': {(1,1): 1} # C11
            }
        ]
        
        c = c_offset
        for b in range(3):
            for term in strassen_terms:
                # Fill U
                for (li, lj), val in term['U'].items():
                    gi = i_offset + li
                    gj = j_offset + lj
                    U[gi * m + gj, c] = val
                
                # Fill V
                for (lj, lk), val in term['V'].items():
                    gj = j_offset + lj
                    gk = 2 * b + lk
                    V[gj * p + gk, c] = val
                
                # Fill W
                for (lk, li), val in term['W'].items():
                    gk = 2 * b + lk
                    gi = i_offset + li
                    W[gk * n + gi, c] = val
                    
                c += 1
                
    # Execute fills
    # Block 1: i=0
    # 1.1: j=0,1
    fill_m126(0, 0, 0)
    # 1.2: j=2,3
    fill_m126(0, 2, 12)
    
    # Block 2: i=1,2
    # 2.1: j=0,1
    fill_m226(1, 0, 24)
    # 2.2: j=2,3
    fill_m226(1, 2, 45)
    
    return {"U": U, "V": V, "W": W}
