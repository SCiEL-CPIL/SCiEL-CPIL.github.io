import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen patterns for 2x2x2
    # u,v are A indices, v,w are B indices, u,w are C indices
    # Patterns: list of (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
    # Indices are (u,v) for A, (v,w) for B, (u,w) for C
    
    patterns = [
        # M1
        ([(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1]),
        # M2
        ([(1,0), (1,1)], [1, 1], [(0,0)], [1], [(1,0), (1,1)], [1, -1]),
        # M3
        ([(0,0)], [1], [(1,0), (1,1)], [1, -1], [(0,1), (1,1)], [1, 1]),
        # M4
        ([(1,1)], [1], [(0,1), (0,0)], [1, -1], [(0,0), (1,0)], [1, 1]),
        # M5
        ([(0,0), (0,1)], [1, 1], [(1,1)], [1], [(0,0), (0,1)], [-1, 1]),
        # M6
        ([(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1], [(1,1)], [1]),
        # M7
        ([(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1], [(0,0)], [1]),
    ]
    
    col = 0
    # i_start is always 0 because n=2
    # j_start in {0, 2}
    # k_start in {0, 2, 4, 6}
    for j_start in [0, 2]:
        for k_start in [0, 2, 4, 6]:
            i_start = 0
            for pat in patterns:
                U_idx, U_coeff, V_idx, V_coeff, W_idx, W_coeff = pat
                
                for idx, coeff in zip(U_idx, U_coeff):
                    u, v = idx
                    gi = i_start + u
                    gj = j_start + v
                    U[gi * m + gj, col] = coeff
                    
                for idx, coeff in zip(V_idx, V_coeff):
                    v, w = idx
                    gj = j_start + v
                    gk = k_start + w
                    V[gj * p + gk, col] = coeff
                    
                for idx, coeff in zip(W_idx, W_coeff):
                    u, w = idx
                    gi = i_start + u
                    gk = k_start + w
                    W[gk * n + gi, col] = coeff
                    
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
