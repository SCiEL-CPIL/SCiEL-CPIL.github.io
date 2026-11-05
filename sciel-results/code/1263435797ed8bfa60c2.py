import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    U_cols = []
    V_cols = []
    W_cols = []

    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Define the 7 Strassen factors for 2x2 matrix multiplication
    # Indices are 0..3 for 2x2 (flattened)
    # u, v, w are lists of length 4
    strassen_2x2 = [
        # M1 = (A00+A11)(B00+B11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (A10+A11)B00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]), # Wait, W is C10 - C00?
        # Let's use the standard reconstruction to define W
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's re-verify the standard factors.
        # M1 = (A00+A11)(B00+B11) -> U: 1001, V: 1001, W: 1111 (coeffs for C00,C01,C10,C11)
        # No, W is a vector in the output space.
        # The factor is u * A * v * B * w * C.
        # So w has coefficients for the output blocks.
        
        # M1: U=[1,0,0,1], V=[1,0,0,1], W=[1, 0, 0, 1] (C00 += M1, C11 += M1)
        # M2: U=[0,0,1,1], V=[1,0,0,0], W=[0, 0, 1, 0] (C10 += M2)
        # M3: U=[1,0,0,0], V=[0,1,-1,0], W=[0, 1, 0, 0] (C01 += M3)
        # M4: U=[0,0,0,1], V=[-1,0,1,0], W=[1, 0, 0, 0] (C00 += M4) -> Wait, C10 += M4 too?
        # Let's check C10 = M2 + M4.
        # M2: C10 += 1. M4: C10 += 1.
        # So W for M4 should be [1, 0, 1, 0]?
        # Let's re-derive or use a known set.
        
        # Known set (from Wikipedia/standard references):
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So W vectors (coeffs for C00, C01, C10, C11):
        # M1: [1, 0, 0, 1]
        # M2: [0, 0, 1, -1]
        # M3: [0, 1, 0, 1]
        # M4: [1, 0, 1, 0]
        # M5: [-1, 1, 0, 0]
        # M6: [0, 0, 0, 1]
        # M7: [1, 0, 0, 0]
        
        # U vectors (coeffs for A00, A01, A10, A11):
        # M1: [1, 0, 0, 1]
        # M2: [0, 0, 1, 1]
        # M3: [1, 0, 0, 0]
        # M4: [0, 0, 0, 1]
        # M5: [1, 1, 0, 0]
        # M6: [-1, 0, 1, 0]
        # M7: [0, 1, 0, -1]
        
        # V vectors (coeffs for B00, B01, B10, B11):
        # M1: [1, 0, 0, 1]
        # M2: [1, 0, 0, 0]
        # M3: [0, 1, 0, -1]
        # M4: [-1, 0, 1, 0]
        # M5: [0, 0, 0, 1]
        # M6: [1, 1, 0, 0]
        # M7: [0, 0, 1, 1]
    ]
    
    # Corrected factors
    strassen_factors = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    # Generate rank 49 factors for 4x4x4
    for r in range(7):
        u_r, v_r, w_r = strassen_factors[r]
        for s in range(7):
            u_s, v_s, w_s = strassen_factors[s]
            
            u_vec = [0]*25
            v_vec = [0]*25
            w_vec = [0]*25
            
            # Fill the 4x4 part (indices 0..15)
            # Index in 4x4 is 4*i_r + i_s
            # u_r is for outer 2x2, u_s is for inner 2x2
            for i_r in range(4):
                if u_r[i_r] != 0:
                    for i_s in range(4):
                        if u_s[i_s] != 0:
                            idx = 4 * i_r + i_s
                            u_vec[idx] += u_r[i_r] * u_s[i_s]
                            
            for i_r in range(4):
                if v_r[i_r] != 0:
                    for i_s in range(4):
                        if v_s[i_s] != 0:
                            idx = 4 * i_r + i_s
                            v_vec[idx] += v_r[i_r] * v_s[i_s]
                            
            for i_r in range(4):
                if w_r[i_r] != 0:
                    for i_s in range(4):
                        if w_s[i_s] != 0:
                            idx = 4 * i_r + i_s
                            w_vec[idx] += w_r[i_r] * w_s[i_s]
                            
            add_col(u_vec, v_vec, w_vec)

    # Add schoolbook for the 1x1 part (index 4)
    # Uncovered: i=4 or j=4 or k=4
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if i == 4 or j == 4 or k == 4:
                    u_vec = [0]*25
                    v_vec = [0]*25
                    w_vec = [0]*25
                    u_vec[i*5 + j] = 1
                    v_vec[j*5 + k] = 1
                    w_vec[k*5 + i] = 1
                    add_col(u_vec, v_vec, w_vec)

    rank = len(U_cols)
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for col_idx in range(rank):
        u_col = U_cols[col_idx]
        v_col = V_cols[col_idx]
        w_col = W_cols[col_idx]
        for row_idx in range(25):
            U[row_idx][col_idx] = u_col[row_idx]
            V[row_idx][col_idx] = v_col[row_idx]
            W[row_idx][col_idx] = w_col[row_idx]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
