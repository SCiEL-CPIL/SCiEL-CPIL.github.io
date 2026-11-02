import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 42
    rank = 42
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    # Strassen vectors for 2x2x2
    # u: A coeffs [A00, A01, A10, A11]
    # v: B coeffs [B00, B01, B10, B11]
    # w: C coeffs [C00, C01, C10, C11]
    # Note: w vectors are derived from C = M1 + ...
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    strassen_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    strassen_w = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 0, 1, 0], # M2: C10
        [0, 1, 0, 0], # M3: C01
        [1, 0, 1, 0], # M4: C00, C10
        [-1, 1, 0, 1],# M5: -C00, C01, C11
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 1]  # M7: C00, C11
    ]
    
    rank_idx = 0
    
    # Loop over the two halves of the middle dimension (m=4 split into 2+2)
    # s=0: A uses cols 0,1; B uses rows 0,1
    # s=1: A uses cols 2,3; B uses rows 2,3
    for s in range(2):
        # Loop over the three blocks of the output dimension (p=6 split into 2+2+2)
        # t=0: K uses cols 0,1
        # t=1: K uses cols 2,3
        # t=2: K uses cols 4,5
        for t in range(3):
            # Loop over 7 Strassen steps
            for l in range(7):
                u_vec = strassen_u[l]
                v_vec = strassen_v[l]
                w_vec = strassen_w[l]
                
                # Update U (A part)
                # A is 2x2 in this block. Local indices i (0,1), j' (0,1).
                # Global A index: i * m + (s * 2 + j')
                for i in range(2):
                    for jp in range(2):
                        global_j = s * 2 + jp
                        u_idx = i * m + global_j
                        u_val = u_vec[i * 2 + jp] # Row-major local A
                        if u_val != 0:
                            U[u_idx][rank_idx] += u_val
                
                # Update V (B part)
                # B is 2x2 in this block. Local indices j' (0,1), k' (0,1).
                # Global B index: (s * 2 + j') * p + (t * 2 + k')
                for jp in range(2):
                    for kp in range(2):
                        global_j = s * 2 + jp
                        global_k = t * 2 + kp
                        v_idx = global_j * p + global_k
                        v_val = v_vec[jp * 2 + kp] # Row-major local B
                        if v_val != 0:
                            V[v_idx][rank_idx] += v_val
                
                # Update W (C part)
                # C is 2x2 in this block. Local indices i (0,1), k' (0,1).
                # Global C index: (t * 2 + k') * n + i
                for i in range(2):
                    for kp in range(2):
                        global_k = t * 2 + kp
                        w_idx = global_k * n + i
                        w_val = w_vec[i * 2 + kp] # Row-major local C
                        if w_val != 0:
                            W[w_idx][rank_idx] += w_val
                
                rank_idx += 1
                
    return {"U": U, "V": V, "W": W}
