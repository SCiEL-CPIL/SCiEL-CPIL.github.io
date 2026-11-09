import time
import numpy as np

def construct():
    n, m, p = 4, 4, 5
    rank = 65
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Block definitions for A (4x4)
    # A00: rows 0,1 cols 0,1
    # A01: rows 0,1 cols 2,3
    # A10: rows 2,3 cols 0,1
    # A11: rows 2,3 cols 2,3
    A_blocks = {
        0: (0, 0), # A00
        1: (0, 2), # A01
        2: (2, 0), # A10
        3: (2, 2)  # A11
    }
    
    # Block definitions for B (4x4 part)
    # B00: rows 0,1 cols 0,1
    # B01: rows 0,1 cols 2,3
    # B10: rows 2,3 cols 0,1
    # B11: rows 2,3 cols 2,3
    B_blocks = {
        0: (0, 0), # B00
        1: (0, 2), # B01
        2: (2, 0), # B10
        3: (2, 2)  # B11
    }
    
    # Output blocks for C (4x4 part)
    # C00: rows 0,1 cols 0,1
    # C01: rows 0,1 cols 2,3
    # C10: rows 2,3 cols 0,1
    # C11: rows 2,3 cols 2,3
    C_blocks = {
        0: (0, 0), # C00
        1: (0, 2), # C01
        2: (2, 0), # C10
        3: (2, 2)  # C11
    }
    
    # Strassen definitions
    # s: 1..7
    # Q_blocks: list of (block_id, sign)
    # R_blocks: list of (block_id, sign)
    # Output: dict of {C_block_id: sign}
    strassen_defs = [
        { # M1
            'Q': [(0, 1), (3, 1)], # A00 + A11
            'R': [(0, 1), (3, 1)], # B00 + B11
            'C': {0: 1, 3: 1}      # C00 + C11
        },
        { # M2
            'Q': [(2, 1), (3, 1)], # A10 + A11
            'R': [(0, 1)],         # B00
            'C': {2: 1}            # C10
        },
        { # M3
            'Q': [(0, 1)],         # A00
            'R': [(1, 1), (3, -1)],# B01 - B11
            'C': {1: 1, 3: -1}     # C01 - C11
        },
        { # M4
            'Q': [(3, 1)],         # A11
            'R': [(2, 1), (0, -1)],# B10 - B00
            'C': {0: -1, 2: 1}     # -C00 + C10
        },
        { # M5
            'Q': [(0, 1), (1, 1)], # A00 + A01
            'R': [(3, 1)],         # B11
            'C': {1: 1, 3: 1}      # C01 + C11
        },
        { # M6
            'Q': [(2, 1), (0, -1)],# A10 - A00
            'R': [(0, 1), (1, 1)], # B00 + B01
            'C': {0: 1}            # C00
        },
        { # M7
            'Q': [(1, 1), (3, -1)],# A01 - A11
            'R': [(2, 1), (3, 1)], # B10 + B11
            'C': {0: -1, 3: 1}     # -C00 + C11
        }
    ]
    
    # Scalar Strassen coefficients c[u,v,t]
    # t=0..6
    # c[u,v,t] is coeff of S_t in P[u,v]
    # P00 = S1 - S4 + S6 - S7
    # P01 = S3 + S5
    # P10 = S2 + S4
    # P11 = S1 - S3 + S5 + S7
    # Indices: S1 is t=0, S2 is t=1, etc.
    c_coeffs = np.zeros((2, 2, 7))
    c_coeffs[0, 0, 0] = 1; c_coeffs[0, 0, 3] = -1; c_coeffs[0, 0, 5] = 1; c_coeffs[0, 0, 6] = -1
    c_coeffs[0, 1, 2] = 1; c_coeffs[0, 1, 4] = 1
    c_coeffs[1, 0, 1] = 1; c_coeffs[1, 0, 3] = 1
    c_coeffs[1, 1, 0] = 1; c_coeffs[1, 1, 2] = -1; c_coeffs[1, 1, 4] = 1; c_coeffs[1, 1, 6] = 1
    
    # Scalar Strassen linear forms
    # For each t, define which Q entries and R entries are used
    # t=0: Q00+Q11, R00+R11
    # t=1: Q10+Q11, R00
    # t=2: Q00, R01-R11
    # t=3: Q11, R10-R00
    # t=4: Q00+Q01, R11
    # t=5: Q10-Q00, R00+R01
    # t=6: Q01-Q11, R10+R11
    
    q_forms = [
        [(0,0,1), (1,1,1)], # t=0
        [(1,0,1), (1,1,1)], # t=1
        [(0,0,1)],          # t=2
        [(1,1,1)],          # t=3
        [(0,0,1), (0,1,1)], # t=4
        [(1,0,1), (0,0,-1)],# t=5
        [(0,1,1), (1,1,-1)] # t=6
    ]
    r_forms = [
        [(0,0,1), (1,1,1)], # t=0
        [(0,0,1)],          # t=1
        [(0,1,1), (1,1,-1)],# t=2
        [(1,0,1), (0,0,-1)],# t=3
        [(1,1,1)],          # t=4
        [(0,0,1), (0,1,1)], # t=5
        [(1,0,1), (1,1,1)]  # t=6
    ]
    
    for s_idx in range(7):
        s = s_idx + 1
        def_s = strassen_defs[s_idx]
        Q_blocks = def_s['Q']
        R_blocks = def_s['R']
        C_outputs = def_s['C']
        
        for t in range(7):
            r = s_idx * 7 + t
            
            # Build U and V
            # U[i,j,r] += coeff
            # V[j,k,r] += coeff
            
            # L_Q = sum of Q entries
            # Each Q entry is a linear combo of A blocks
            # Q_entry(u,v) = sum_{b, sign} sign * A_block[b][u,v]
            
            # We need to accumulate coefficients for each A[i,j]
            U_coeffs = np.zeros((4, 4))
            V_coeffs = np.zeros((4, 4)) # Only first 4 cols of B
            
            for (u, v, sign) in q_forms[t]:
                for (b_id, b_sign) in Q_blocks:
                    r0, c0 = A_blocks[b_id]
                    # A_block[b_id][u,v] is A[r0+u, c0+v]
                    i = r0 + u
                    j = c0 + v
                    U_coeffs[i, j] += sign * b_sign
            
            for (u, v, sign) in r_forms[t]:
                for (b_id, b_sign) in R_blocks:
                    r0, c0 = B_blocks[b_id]
                    # B_block[b_id][u,v] is B[r0+u, c0+v]
                    j = r0 + u
                    k = c0 + v
                    V_coeffs[j, k] += sign * b_sign
            
            # Fill U and V
            for i in range(4):
                for j in range(4):
                    if U_coeffs[i, j] != 0:
                        U[i*m + j, r] = U_coeffs[i, j]
            
            for j in range(4):
                for k in range(4):
                    if V_coeffs[j, k] != 0:
                        V[j*p + k, r] = V_coeffs[j, k]
            
            # Fill W
            # For each output block C_b with sign sigma
            for c_id, sigma in C_outputs.items():
                r0, c0 = C_blocks[c_id]
                for u in range(2):
                    for v in range(2):
                        coeff = c_coeffs[u, v, t] * sigma
                        if coeff != 0:
                            i = r0 + u
                            k = c0 + v
                            W[k*n + i, r] = coeff
    
    # Add 16 scalar terms for the 5th column
    for i in range(4):
        for j in range(4):
            r = 49 + i*4 + j
            U[i*m + j, r] = 1
            V[j*p + 4, r] = 1
            W[4*n + i, r] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
