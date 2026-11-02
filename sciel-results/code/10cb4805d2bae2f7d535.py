import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    rank = 42
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def add_strassen_block(A_idx, B_idx, C_idx, col_start):
        # A_idx: [a, b, c, d] for A = [[a, b], [c, d]]
        # B_idx: [e, f, g, h] for B = [[e, f], [g, h]]
        # C_idx: [c00, c01, c10, c11] for C = [[c00, c01], [c10, c11]]
        a, b, c, d = A_idx
        e, f, g, h = B_idx
        c00, c01, c10, c11 = C_idx
        
        col = col_start
        
        # M1 = (a+d)(e+h)
        U[a, col] = 1; U[d, col] = 1
        V[e, col] = 1; V[h, col] = 1
        W[c00, col] = 1
        W[c11, col] = 1
        col += 1
        
        # M2 = (c+d)e
        U[c, col] = 1; U[d, col] = 1
        V[e, col] = 1
        W[c10, col] = 1
        W[c11, col] = -1
        col += 1
        
        # M3 = a(f-h)
        U[a, col] = 1
        V[f, col] = 1; V[h, col] = -1
        W[c01, col] = 1
        W[c11, col] = 1
        col += 1
        
        # M4 = d(g-e)
        U[d, col] = 1
        V[g, col] = 1; V[e, col] = -1
        W[c00, col] = 1
        W[c10, col] = 1
        col += 1
        
        # M5 = (a+b)h
        U[a, col] = 1; U[b, col] = 1
        V[h, col] = 1
        W[c00, col] = -1
        W[c01, col] = 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c, col] = 1; U[a, col] = -1
        V[e, col] = 1; V[f, col] = 1
        W[c11, col] = 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b, col] = 1; U[d, col] = -1
        V[g, col] = 1; V[h, col] = 1
        W[c00, col] = 1
        col += 1
        
        return col

    # Block indices
    # A_L: [[a00, a01], [a10, a11]] -> [0, 1, 4, 5]
    A_L = [0, 1, 4, 5]
    # A_R: [[a02, a03], [a12, a13]] -> [2, 3, 6, 7]
    A_R = [2, 3, 6, 7]
    
    # B_T0: [[b00, b01], [b10, b11]] -> [0, 1, 6, 7]
    B_T0 = [0, 1, 6, 7]
    # B_T1: [[b02, b03], [b12, b13]] -> [2, 3, 8, 9]
    B_T1 = [2, 3, 8, 9]
    # B_T2: [[b04, b05], [b14, b15]] -> [4, 5, 10, 11]
    B_T2 = [4, 5, 10, 11]
    
    # B_B0: [[b20, b21], [b30, b31]] -> [12, 13, 18, 19]
    B_B0 = [12, 13, 18, 19]
    # B_B1: [[b22, b23], [b32, b33]] -> [14, 15, 20, 21]
    B_B1 = [14, 15, 20, 21]
    # B_B2: [[b24, b25], [b34, b35]] -> [16, 17, 22, 23]
    B_B2 = [16, 17, 22, 23]
    
    # C blocks
    # C_L0: [[c00, c01], [c10, c11]] -> [0, 1, 6, 7]
    C_L0 = [0, 1, 6, 7]
    # C_L1: [[c02, c03], [c12, c13]] -> [2, 3, 8, 9]
    C_L1 = [2, 3, 8, 9]
    # C_L2: [[c04, c05], [c14, c15]] -> [4, 5, 10, 11]
    C_L2 = [4, 5, 10, 11]
    
    # C_R0: [[c00, c01], [c10, c11]] -> [0, 1, 6, 7]
    C_R0 = [0, 1, 6, 7]
    # C_R1: [[c02, c03], [c12, c13]] -> [2, 3, 8, 9]
    C_R1 = [2, 3, 8, 9]
    # C_R2: [[c04, c05], [c14, c15]] -> [4, 5, 10, 11]
    C_R2 = [4, 5, 10, 11]
    
    col = 0
    col = add_strassen_block(A_L, B_T0, C_L0, col)
    col = add_strassen_block(A_L, B_T1, C_L1, col)
    col = add_strassen_block(A_L, B_T2, C_L2, col)
    col = add_strassen_block(A_R, B_B0, C_R0, col)
    col = add_strassen_block(A_R, B_B1, C_R1, col)
    col = add_strassen_block(A_R, B_B2, C_R2, col)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
