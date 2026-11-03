import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    blocks = [
        (0, 0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 2, 0, 2),
    ]
    
    col = 0
    for (ar, ac, br, bc, cr, cc) in blocks:
        for (u_loc, v_loc, w_loc) in strassen_terms:
            if time.time() > deadline:
                break
                
            for lr in range(2):
                for lc in range(2):
                    coeff_u = u_loc[lr * 2 + lc]
                    if coeff_u != 0:
                        global_i = ar + lr
                        global_j = ac + lc
                        u_idx = global_i * m + global_j
                        U[u_idx][col] += coeff_u
                        
            for lr in range(2):
                for lc in range(2):
                    coeff_v = v_loc[lr * 2 + lc]
                    if coeff_v != 0:
                        global_j = br + lr
                        global_k = bc + lc
                        v_idx = global_j * p + global_k
                        V[v_idx][col] += coeff_v
                        
            for lr in range(2):
                for lc in range(2):
                    coeff_w = w_loc[lr * 2 + lc]
                    if coeff_w != 0:
                        global_i = cr + lr
                        global_k = cc + lc
                        w_idx = global_k * n + global_i
                        W[w_idx][col] += coeff_w
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
