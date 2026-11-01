import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6
    
    # Rank 42 decomposition
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Define the 7 Strassen factors for 2x2x2
    # U_part: dict {A_index: coeff}
    # V_part: dict {B_index: coeff}
    # W_part: dict {C_index: coeff}
    
    # Indices: A: 0,1,2,3 (i*2+j)
    #          B: 0,1,2,3 (j*2+k)
    #          C: 0,1,2,3 (i*2+k) -> Wait, W is (p*n, rank). Index is k*n+i.
    # In the previous code, W index was calculated as `c = (kp * 2 + ll) * n + li`.
    # Here, local C index `loc_idx` was `li * 2 + ll`.
    # So `li` is row i, `ll` is col k.
    # Global W index `k*n+i`.
    
    s_factors = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    col = 0
    for jm in range(2):
        for kp in range(3):
            for s in s_factors:
                u_d, v_d, w_d = s
                for a_idx, val in u_d.items():
                    li, lj = divmod(a_idx, 2)
                    U[li * m + (jm * 2 + lj), col] = val
                for b_idx, val in v_d.items():
                    lj, lk = divmod(b_idx, 2)
                    V[(jm * 2 + lj) * p + (kp * 2 + lk), col] = val
                for c_idx, val in w_d.items():
                    li, lk = divmod(c_idx, 2)
                    W[(kp * 2 + lk) * n + li, col] = val
                col += 1
                
    if time.time() - start_time > 110:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
