import numpy as np
import time

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_sp, v_sp, w_sp):
        for idx, val in u_sp:
            U[idx][col] += val
        for idx, val in v_sp:
            V[idx][col] += val
        for idx, val in w_sp:
            W[idx][col] += val

    # Block 1 terms (cols 0-6)
    # Term 0 (M1)
    add_term(0,
             [(0, 1), (3, 1)],
             [(0, 1), (5, 1)],
             [(0, 1), (3, 1)])
    
    # Term 1 (M2)
    add_term(1,
             [(2, 1), (3, 1)],
             [(0, 1)],
             [(1, 1)])
    
    # Term 2 (M3)
    add_term(2,
             [(0, 1)],
             [(1, 1), (5, -1)],
             [(2, 1)])
    
    # Term 3 (M4)
    add_term(3,
             [(3, 1)],
             [(4, 1), (0, -1)],
             [(0, 1), (1, 1)])
    
    # Term 4 (M5)
    add_term(4,
             [(0, 1), (1, 1)],
             [(5, 1)],
             [(0, -1), (2, 1)])
    
    # Term 5 (M6)
    add_term(5,
             [(1, 1), (3, -1)],
             [(4, 1), (5, 1)],
             [(3, 1)])
    
    # Term 6 (M7)
    add_term(6,
             [(0, 1), (2, -1)],
             [(0, 1), (1, 1)],
             [(0, 1)])

    # Block 2 terms (cols 7-13)
    # Term 7 (M1')
    add_term(7,
             [(0, 1), (3, 1)],
             [(2, 1), (7, 1)],
             [(4, 1), (7, 1)])
    
    # Term 8 (M2')
    add_term(8,
             [(2, 1), (3, 1)],
             [(2, 1)],
             [(5, 1)])
    
    # Term 9 (M3')
    add_term(9,
             [(0, 1)],
             [(3, 1), (7, -1)],
             [(6, 1)])
    
    # Term 10 (M4')
    add_term(10,
             [(3, 1)],
             [(6, 1), (2, -1)],
             [(4, 1), (5, 1)])
    
    # Term 11 (M5')
    add_term(11,
             [(0, 1), (1, 1)],
             [(7, 1)],
             [(4, -1), (6, 1)])
    
    # Term 12 (M6')
    add_term(12,
             [(1, 1), (3, -1)],
             [(6, 1), (7, 1)],
             [(7, 1)])
    
    # Term 13 (M7')
    add_term(13,
             [(0, 1), (2, -1)],
             [(2, 1), (3, 1)],
             [(4, 1)])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
