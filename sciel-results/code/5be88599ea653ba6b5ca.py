import numpy as np
import time

def construct():
    """
    Rank-28 Strassen-based decomposition for [2,4,4] matrix multiplication.
    
    Uses 4 independent 2x2x2 Strassen blocks (rank 7 each).
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i
    
    blocks = [
        (0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0),
        (0, 0, 0, 2, 2),
        (0, 2, 2, 2, 2)
    ]
    
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]
    
    current_rank = 0
    
    for (a_r, a_c, b_r, b_c, c_c) in blocks:
        u_map = [
            u_idx(a_r, a_c),
            u_idx(a_r, a_c + 1),
            u_idx(a_r + 1, a_c),
            u_idx(a_r + 1, a_c + 1)
        ]
        
        v_map = [
            v_idx(b_r, b_c),
            v_idx(b_r, b_c + 1),
            v_idx(b_r + 1, b_c),
            v_idx(b_r + 1, b_c + 1)
        ]
        
        w_map = [
            w_idx(0, c_c),
            w_idx(0, c_c + 1),
            w_idx(1, c_c),
            w_idx(1, c_c + 1)
        ]
        
        for (u_terms, v_terms, w_terms) in strassen_terms:
            for local_idx, coeff in u_terms:
                U[u_map[local_idx], current_rank] += coeff
            for local_idx, coeff in v_terms:
                V[v_map[local_idx], current_rank] += coeff
            for local_idx, coeff in w_terms:
                W[w_map[local_idx], current_rank] += coeff
            current_rank += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
