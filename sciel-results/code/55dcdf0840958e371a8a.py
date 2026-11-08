import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def add_block(offset_j, offset_k, col_offset):
        terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
        ]
        
        for t_idx, (la, lb, wc) in enumerate(terms):
            col = col_offset + t_idx
            
            for (li, lj), coeff in la.items():
                gi = li
                gj = offset_j + lj
                U[gi * m + gj, col] += coeff
                
            for (lj, lk), coeff in lb.items():
                gj = offset_j + lj
                gk = offset_k + lk
                V[gj * p + gk, col] += coeff
                
            for (li, lk), coeff in wc.items():
                gi = li
                gk = offset_k + lk
                W[gk * n + gi, col] += coeff

    add_block(0, 0, 0)
    add_block(0, 2, 7)
    add_block(2, 4, 14)
    add_block(2, 6, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
