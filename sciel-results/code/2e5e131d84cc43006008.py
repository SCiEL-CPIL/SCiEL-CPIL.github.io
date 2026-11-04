import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We will build the columns one by one
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vals, v_vals, w_vals):
        """
        u_vals: dict mapping (i, j) to value, or list of length n*m
        v_vals: dict mapping (j, k) to value, or list of length m*p
        w_vals: dict mapping (k, i) to value, or list of length p*n
        """
        U_row = [0.0] * (n * m)
        V_row = [0.0] * (m * p)
        W_row = [0.0] * (p * n)
        
        # Fill U
        if isinstance(u_vals, dict):
            for (i, j), val in u_vals.items():
                U_row[i * m + j] = val
        else:
            for idx, val in enumerate(u_vals):
                if val != 0:
                    U_row[idx] = val
                    
        # Fill V
        if isinstance(v_vals, dict):
            for (j, k), val in v_vals.items():
                V_row[j * p + k] = val
        else:
            for idx, val in enumerate(v_vals):
                if val != 0:
                    V_row[idx] = val
                    
        # Fill W
        if isinstance(w_vals, dict):
            for (k, i), val in w_vals.items():
                W_row[k * n + i] = val
        else:
            for idx, val in enumerate(w_vals):
                if val != 0:
                    W_row[idx] = val
                    
        U_cols.append(U_row)
        V_cols.append(V_row)
        W_cols.append(W_row)

    def add_strassen_2x2x2(i_offset, j_offset, k_offset):
        """
        Adds Strassen decomposition for A[i_offset:i_offset+2, j_offset:j_offset+2] * B[j_offset:j_offset+2, k_offset:k_offset+2]
        """
        # A indices: i in [i_off, i_off+1], j in [j_off, j_off+1]
        # B indices: j in [j_off, j_off+1], k in [k_off, k_off+1]
        
        # M1 = (A11 + A22)(B11 + B22)
        u1 = {(i_offset, j_offset): 1, (i_offset+1, j_offset+1): 1}
        v1 = {(j_offset, k_offset): 1, (j_offset+1, k_offset+1): 1}
        w1 = {(k_offset, i_offset): 1, (k_offset+1, i_offset): 1, (k_offset, i_offset+1): 1, (k_offset+1, i_offset+1): 1}
        # C11 += M1, C12 += 0, C21 += 0, C22 += M1
        # W mapping: W[k*n+i]
        # M1 contributes to C11 (k=k_off, i=i_off) and C22 (k=k_off+1, i=i_off+1)?
        # Wait. C11 is at (i_off, k_off). C22 is at (i_off+1, k_off+1).
        # Strassen formulas:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M6
        # C22 = M1 - M3 + M6 + M7
        
        # Let's define the W coefficients for each M.
        # M1: C11 (+1), C22 (+1)
        # M2: C21 (+1)
        # M3: C12 (+1), C22 (-1)
        # M4: C11 (+1)
        # M5: C12 (+1), C11 (-1)
        # M6: C21 (+1), C22 (+1)
        # M7: C11 (+1), C22 (+1)
        
        # We need to add 7 terms.
        
        # Term 1: M1
        u_m1 = {(i_offset, j_offset): 1, (i_offset+1, j_offset+1): 1}
        v_m1 = {(j_offset, k_offset): 1, (j_offset+1, k_offset+1): 1}
        w_m1 = {(k_offset, i_offset): 1, (k_offset+1, i_offset+1): 1}
        add_term(u_m1, v_m1, w_m1)
        
        # Term 2: M2
        u_m2 = {(i_offset+1, j_offset): 1, (i_offset+1, j_offset+1): 1}
        v_m2 = {(j_offset, k_offset): 1}
        w_m2 = {(k_offset, i_offset+1): 1}
        add_term(u_m2, v_m2, w_m2)
        
        # Term 3: M3
        u_m3 = {(i_offset, j_offset): 1}
        v_m3 = {(j_offset+1, k_offset): 1, (j_offset+1, k_offset+1): -1}
        w_m3 = {(k_offset, i_offset): 1, (k_offset+1, i_offset+1): -1}
        add_term(u_m3, v_m3, w_m3)
        
        # Term 4: M4
        u_m4 = {(i_offset+1, j_offset+1): 1}
        v_m4 = {(j_offset, k_offset+1): 1, (j_offset, k_offset): -1}
        w_m4 = {(k_offset, i_offset): 1}
        add_term(u_m4, v_m4, w_m4)
        
        # Term 5: M5
        u_m5 = {(i_offset, j_offset): 1, (i_offset, j_offset+1): 1}
        v_m5 = {(j_offset+1, k_offset+1): 1}
        w_m5 = {(k_offset+1, i_offset): 1, (k_offset, i_offset): -1}
        add_term(u_m5, v_m5, w_m5)
        
        # Term 6: M6
        u_m6 = {(i_offset+1, j_offset): 1, (i_offset, j_offset): -1}
        v_m6 = {(j_offset, k_offset): 1, (j_offset, k_offset+1): 1}
        w_m6 = {(k_offset, i_offset+1): 1, (k_offset+1, i_offset+1): 1}
        add_term(u_m6, v_m6, w_m6)
        
        # Term 7: M7
        u_m7 = {(i_offset, j_offset+1): 1, (i_offset+1, j_offset+1): -1}
        v_m7 = {(j_offset+1, k_offset): 1, (j_offset+1, k_offset+1): 1}
        w_m7 = {(k_offset, i_offset): 1, (k_offset+1, i_offset+1): 1}
        add_term(u_m7, v_m7, w_m7)

    def add_outer_product_2x1x2(i_offset, j_idx, k_offset):
        """
        Adds decomposition for A[i_offset:i_offset+2, j_idx] * B[j_idx, k_offset:k_offset+2]
        A is 2x1, B is 1x2.
        C is 2x2.
        C_ik = A_i,j_idx * B_j_idx,k
        """
        u_op = {(i_offset, j_idx): 1, (i_offset+1, j_idx): 1}
        v_op = {(j_idx, k_offset): 1, (j_idx, k_offset+1): 1}
        w_op = {(k_offset, i_offset): 1, (k_offset+1, i_offset): 1, (k_offset, i_offset+1): 1, (k_offset+1, i_offset+1): 1}
        # Wait, W should be 1 for all combinations?
        # Sum_c U_ic V_ck W_ki
        # If U_ic = A_i,j, V_ck = B_j,k, W_ki = 1.
        # Then Sum A_i,j B_j,k * 1 = C_ik.
        # Yes.
        add_term(u_op, v_op, w_op)

    # Part 1: 2x2x6 (j in 0,1)
    # Split k into 0,1; 2,3; 4,5
    for k_block in range(3):
        k_off = k_block * 2
        add_strassen_2x2x2(0, 0, k_off)

    # Part 2: 2x3x6 (j in 2,3,4)
    # Split k into 0,1; 2,3; 4,5
    for k_block in range(3):
        k_off = k_block * 2
        # j=2 part: 2x1x2 outer product
        add_outer_product_2x1x2(0, 2, k_off)
        # j=3,4 part: 2x2x2 Strassen
        add_strassen_2x2x2(0, 3, k_off)

    # Convert to matrices
    rank = len(U_cols)
    U = [U_cols[c][r] for r in range(n*m) for c in range(rank)] # Wait, need to transpose
    # U is (n*m, rank). U_cols is list of rank vectors of length n*m.
    # So U[i][c] = U_cols[c][i]
    U = [[U_cols[c][i] for c in range(rank)] for i in range(n*m)]
    V = [[V_cols[c][j] for c in range(rank)] for j in range(m*p)]
    W = [[W_cols[c][k] for c in range(rank)] for k in range(p*n)]

    return {"U": U, "V": V, "W": W}
