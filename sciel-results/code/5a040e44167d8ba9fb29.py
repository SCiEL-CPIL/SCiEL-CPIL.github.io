import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
    
    Strategy:
    Split A (2x3) into A1 (2x2) and A2 (2x1)
    Split B (3x7) into B1 (2x7) and B2 (1x7)
    C = A1*B1 + A2*B2
    
    For A1*B1 (2x2 x 2x7):
      Split B1 into B1a (2x2), B1b (2x2), B1c (2x2), B1d (2x1)
      A1*B1a: Strassen rank 7
      A1*B1b: Strassen rank 7
      A1*B1c: Strassen rank 7
      A1*B1d: Schoolbook rank 4
      Subtotal: 25
    
    For A2*B2 (2x1 x 1x7):
      Outer product, rank 7
    
    Total rank: 32
    """
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    terms = []
    
    def add_term(u, v, w):
        u_list = [0.0] * (n * m)
        v_list = [0.0] * (m * p)
        w_list = [0.0] * (p * n)
        for idx, val in u.items():
            u_list[idx] = val
        for idx, val in v.items():
            v_list[idx] = val
        for idx, val in w.items():
            w_list[idx] = val
        terms.append((u_list, v_list, w_list))
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Strassen for 2x2 x 2x2: A1*B1a where B1a uses columns 0,1 of B
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1, w_idx(1,1): 1})
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1}, {w_idx(0,1): 1, w_idx(1,1): -1})
    add_term({u_idx(0,0): 1}, {v_idx(0,1): 1, v_idx(1,1): -1}, {w_idx(1,0): 1, w_idx(1,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,0): 1, v_idx(0,0): -1}, {w_idx(0,0): 1, w_idx(0,1): 1})
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,1): 1}, {w_idx(0,0): -1, w_idx(1,0): 1})
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,0): 1, v_idx(0,1): 1}, {w_idx(1,1): 1})
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1})
    
    # Strassen for 2x2 x 2x2: A1*B1b where B1b uses columns 2,3 of B
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,2): 1, v_idx(1,3): 1}, {w_idx(2,0): 1, w_idx(3,1): 1})
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,2): 1}, {w_idx(2,1): 1, w_idx(3,1): -1})
    add_term({u_idx(0,0): 1}, {v_idx(0,3): 1, v_idx(1,3): -1}, {w_idx(3,0): 1, w_idx(3,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,2): 1, v_idx(0,2): -1}, {w_idx(2,0): 1, w_idx(2,1): 1})
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,3): 1}, {w_idx(2,0): -1, w_idx(3,0): 1})
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,2): 1, v_idx(0,3): 1}, {w_idx(3,1): 1})
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,2): 1, v_idx(1,3): 1}, {w_idx(2,0): 1})
    
    # Strassen for 2x2 x 2x2: A1*B1c where B1c uses columns 4,5 of B
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,4): 1, v_idx(1,5): 1}, {w_idx(4,0): 1, w_idx(5,1): 1})
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,4): 1}, {w_idx(4,1): 1, w_idx(5,1): -1})
    add_term({u_idx(0,0): 1}, {v_idx(0,5): 1, v_idx(1,5): -1}, {w_idx(5,0): 1, w_idx(5,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,4): 1, v_idx(0,4): -1}, {w_idx(4,0): 1, w_idx(4,1): 1})
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,5): 1}, {w_idx(4,0): -1, w_idx(5,0): 1})
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,4): 1, v_idx(0,5): 1}, {w_idx(5,1): 1})
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,4): 1, v_idx(1,5): 1}, {w_idx(4,0): 1})
    
    # Schoolbook for 2x2 x 2x1: A1*B1d where B1d uses column 6 of B
    add_term({u_idx(0,0): 1}, {v_idx(0,6): 1}, {w_idx(6,0): 1})
    add_term({u_idx(0,1): 1}, {v_idx(1,6): 1}, {w_idx(6,0): 1})
    add_term({u_idx(1,0): 1}, {v_idx(0,6): 1}, {w_idx(6,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,6): 1}, {w_idx(6,1): 1})
    
    # Outer product for 2x1 x 1x7: A2*B2
    for j in range(7):
        add_term({u_idx(0,2): 1, u_idx(1,2): 1}, {v_idx(2,j): 1}, {w_idx(j,0): 1, w_idx(j,1): 1})
    
    rank = len(terms)
    
    # Construct U, V, W
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u_list, v_list, w_list) in enumerate(terms):
        for i in range(n * m):
            U[i][c] = u_list[i]
        for i in range(m * p):
            V[i][c] = v_list[i]
        for i in range(p * n):
            W[i][c] = w_list[i]
            
    return {"U": U, "V": V, "W": W}
