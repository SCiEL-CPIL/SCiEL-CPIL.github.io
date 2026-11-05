"""
Program for finding a rank-minimal tensor decomposition of the 3x4 by 4x8 matrix multiplication.
Uses a hybrid strategy: Strassen decomposition for 2x2x2 blocks embedded in 2x4x8, 
and schoolbook for the remaining 1x4x8 part.
Total rank: 2*28 + 32 = 88.
"""
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Initialize lists to hold the columns of U, V, W
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a rank-1 term
    def add_term(u_vals, v_vals, w_vals):
        # u_vals: dict or list of (index, value)
        # v_vals: dict or list of (index, value)
        # w_vals: dict or list of (index, value)
        
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)
        
        for idx, val in u_vals:
            u[idx] = val
        for idx, val in v_vals:
            v[idx] = val
        for idx, val in w_vals:
            w[idx] = val
            
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    # Strassen 2x2x2 factors
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1) -> Note: W[k*n+i]
    # C_00 (p): k=0, i=0 -> 0
    # C_10 (r): k=0, i=1 -> 1
    # C_01 (q): k=1, i=0 -> 2
    # C_11 (s): k=1, i=1 -> 3
    
    strassen_terms = [
        # (U, V, W)
        ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    # Part 1.1: A_1L (2x2) * B_T (2x8)
    # A_1L indices in global U:
    # A[0,0] -> 0
    # A[0,1] -> 1
    # A[1,0] -> 4
    # A[1,1] -> 5
    
    # B_T is 2x8. Split into 4 blocks of 2x2.
    # Block q (0..3) uses B cols 2q, 2q+1 from rows 0,1.
    # B[0, 2q] -> 2q
    # B[0, 2q+1] -> 2q+1
    # B[1, 2q] -> 8+2q
    # B[1, 2q+1] -> 8+2q+1
    
    # Output C_1L (2x8). Block q uses C cols 2q, 2q+1 from rows 0,1.
    # C[0, 2q] -> 2q*3 + 0 = 6q
    # C[1, 2q] -> 2q*3 + 1 = 6q+1
    # C[0, 2q+1] -> (2q+1)*3 + 0 = 6q+3
    # C[1, 2q+1] -> (2q+1)*3 + 1 = 6q+4
    
    a_1l_indices = [0, 1, 4, 5]
    
    for q in range(4):
        # B indices for this block
        b_indices = [2*q, 2*q+1, 8+2*q, 8+2*q+1]
        # W indices for this block
        w_indices = [6*q, 6*q+1, 6*q+3, 6*q+4]
        
        for u_s, v_s, w_s in strassen_terms:
            u_vals = [(a_1l_indices[i], u_s[i]) for i in range(4) if u_s[i] != 0]
            v_vals = [(b_indices[i], v_s[i]) for i in range(4) if v_s[i] != 0]
            w_vals = [(w_indices[i], w_s[i]) for i in range(4) if w_s[i] != 0]
            add_term(u_vals, v_vals, w_vals)
            
    # Part 1.2: A_1R (2x2) * B_B (2x8)
    # A_1R indices in global U:
    # A[0,2] -> 2
    # A[0,3] -> 3
    # A[1,2] -> 6
    # A[1,3] -> 7
    
    # B_B is 2x8. Rows 2,3.
    # B[2, 2q] -> 16+2q
    # B[2, 2q+1] -> 16+2q+1
    # B[3, 2q] -> 24+2q
    # B[3, 2q+1] -> 24+2q+1
    
    # Output C_1R is same C rows 0,1.
    # W indices same as Part 1.1.
    
    a_1r_indices = [2, 3, 6, 7]
    
    for q in range(4):
        b_indices = [16+2*q, 16+2*q+1, 24+2*q, 24+2*q+1]
        w_indices = [6*q, 6*q+1, 6*q+3, 6*q+4]
        
        for u_s, v_s, w_s in strassen_terms:
            u_vals = [(a_1r_indices[i], u_s[i]) for i in range(4) if u_s[i] != 0]
            v_vals = [(b_indices[i], v_s[i]) for i in range(4) if v_s[i] != 0]
            w_vals = [(w_indices[i], w_s[i]) for i in range(4) if w_s[i] != 0]
            add_term(u_vals, v_vals, w_vals)

    # Part 2: A_2 (1x4) * B (4x8)
    # A_2 indices:
    # A[2,0] -> 8
    # A[2,1] -> 9
    # A[2,2] -> 10
    # A[2,3] -> 11
    
    # B is 4x8.
    # B[j, k] -> j*8 + k
    
    # C_2 is row 2 of C.
    # C[2, k] -> k*3 + 2
    
    for j in range(4):
        a_idx = 8 + j
        for k in range(8):
            b_idx = j * 8 + k
            w_idx = k * 3 + 2
            add_term([(a_idx, 1)], [(b_idx, 1)], [(w_idx, 1)])

    # Convert lists of lists to matrices (or just return lists if interface allows)
    # Interface expects 'U', 'V', 'W' which are likely lists of lists or numpy arrays.
    # The seed program returned lists of lists.
    
    rank = len(U_cols)
    U = np.array(U_cols).T.tolist()
    V = np.array(V_cols).T.tolist()
    W = np.array(W_cols).T.tolist()
    
    return {"U": U, "V": V, "W": W}
