import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen coefficients
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: j=0,1; k=0..7
    # C indices: i=0,1; k=0..7
    
    # We define the 28 terms explicitly as (u_idx, v_idx, w_idx, value)
    # But since it's block-wise, we can generate them.
    
    # Strassen block coefficients
    # U_coeffs[s] is a dict of {A_idx: val}
    # V_coeffs[s] is a dict of {B_local_idx: val}
    # W_coeffs[s] is a dict of {C_local_idx: val}
    
    U_c = [
        {0: 1, 3: 1}, {2: 1, 3: 1}, {0: 1}, {3: 1}, {0: 1, 1: 1}, {2: 1, 0: -1}, {1: 1, 3: -1}
    ]
    V_c = [
        {0: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}, {2: 1, 0: -1}, {3: 1}, {0: 1, 1: 1}, {2: 1, 3: 1}
    ]
    W_c = [
        {0: 1, 3: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 1: 1}, {0: -1, 2: 1}, {3: 1}, {0: 1}
    ]
    
    for q in range(4):
        for s in range(7):
            r = q * 7 + s
            for u_idx, val in U_c[s].items():
                U[u_idx, r] = val
            for v_local, val in V_c[s].items():
                j = v_local // 2
                k_loc = v_local % 2
                k = 2*q + k_loc
                v_idx = j * p + k
                V[v_idx, r] = val
            for w_local, val in W_c[s].items():
                i = w_local % 2
                k_loc = w_local // 2
                k = 2*q + k_loc
                w_idx = k * n + i
                W[w_idx, r] = val
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
