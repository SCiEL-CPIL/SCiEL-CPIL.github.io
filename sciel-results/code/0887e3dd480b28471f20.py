"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    # U shape: (n*m, R) = (6, R)
    # V shape: (m*p, R) = (18, R)
    # W shape: (p*n, R) = (12, R)
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    def add_term(u_dict, v_dict, w_dict):
        u = np.zeros(6)
        v = np.zeros(18)
        w = np.zeros(12)
        for idx, val in u_dict.items():
            u[idx] = val
        for idx, val in v_dict.items():
            v[idx] = val
        for idx, val in w_dict.items():
            w[idx] = val
        terms_U.append(u)
        terms_V.append(v)
        terms_W.append(w)
        
    # Part 1: A_12 * B_2 (Outer product of col 2 of A and row 2 of B)
    # A[i, 2] * B[2, k] -> C[k, i]
    for i in range(2):
        for k in range(6):
            u_idx = i * 3 + 2
            v_idx = 2 * 6 + k
            w_idx = k * 2 + i
            add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
            
    # Part 2: A_11 * B_1 (Strassen on 2x2 blocks)
    # A is 2x2 (cols 0,1), B is 2x6 (rows 0,1)
    # Split B into 3 blocks of 2x2
    for j in range(3):
        # B block indices
        b00 = 2*j
        b01 = 2*j + 1
        b10 = 6 + 2*j
        b11 = 6 + 2*j + 1
        
        # C block indices
        c00 = 4*j
        c01 = 4*j + 2
        c10 = 4*j + 1
        c11 = 4*j + 3
        
        # Strassen terms
        # M1
        add_term({0: 1, 4: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1})
        # M2
        add_term({3: 1, 4: 1}, {b00: 1}, {c10: 1, c11: -1})
        # M3
        add_term({0: 1}, {b01: 1, b11: -1}, {c01: 1, c11: 1})
        # M4
        add_term({4: 1}, {b10: 1, b00: -1}, {c00: 1, c10: 1})
        # M5
        add_term({0: 1, 1: 1}, {b11: 1}, {c00: -1, c01: 1})
        # M6
        add_term({3: 1, 0: -1}, {b00: 1, b01: 1}, {c11: 1})
        # M7
        add_term({1: 1, 4: -1}, {b10: 1, b11: 1}, {c00: 1})
        
    R = len(terms_U)
    U = np.array(terms_U).T.tolist()
    V = np.array(terms_V).T.tolist()
    W = np.array(terms_W).T.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
