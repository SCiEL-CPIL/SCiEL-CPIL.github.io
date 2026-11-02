import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_map, v_map, w_map):
        u_col = [0.0] * (n * m)
        v_col = [0.0] * (m * p)
        w_col = [0.0] * (p * n)
        
        for (i, j), c in u_map.items():
            if c != 0:
                u_col[i * m + j] = c
                
        for (j, k), c in v_map.items():
            if c != 0:
                v_col[j * p + k] = c
                
        for (k, i), c in w_map.items():
            if c != 0:
                w_col[k * n + i] = c
                
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    def add_schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u_map = {(i, j): 1.0}
                    v_map = {(j, k): 1.0}
                    w_map = {(k, i): 1.0}
                    add_term(u_map, v_map, w_map)

    def add_strassen_2x2x2(i0, j0, k0):
        # M1
        add_term(
            {(i0, j0): 1.0, (i0+1, j0+1): 1.0},
            {(j0, k0): 1.0, (j0+1, k0+1): 1.0},
            {(k0, i0): 1.0, (k0+1, i0+1): 1.0}
        )
        # M2
        add_term(
            {(i0+1, j0): 1.0, (i0+1, j0+1): 1.0},
            {(j0, k0): 1.0},
            {(k0, i0+1): 1.0}
        )
        # M3
        add_term(
            {(i0, j0): 1.0},
            {(j0, k0+1): 1.0, (j0+1, k0+1): -1.0},
            {(k0+1, i0): 1.0}
        )
        # M4
        add_term(
            {(i0+1, j0+1): 1.0},
            {(j0+1, k0): 1.0, (j0, k0): -1.0},
            {(k0, i0+1): 1.0, (k0+1, i0+1): 1.0}
        )
        # M5
        add_term(
            {(i0, j0): 1.0, (i0, j0+1): 1.0},
            {(j0+1, k0+1): 1.0},
            {(k0+1, i0): -1.0, (k0+1, i0+1): 1.0}
        )
        # M6
        add_term(
            {(i0+1, j0): 1.0, (i0, j0): -1.0},
            {(j0, k0): 1.0, (j0, k0+1): 1.0},
            {(k0+1, i0+1): 1.0}
        )
        # M7
        add_term(
            {(i0, j0+1): 1.0, (i0+1, j0+1): -1.0},
            {(j0+1, k0): 1.0, (j0+1, k0+1): 1.0},
            {(k0, i0): 1.0, (k0, i0+1): 1.0}
        )

    # Construct 3x3x3 part
    add_strassen_2x2x2(0, 0, 0)
    add_schoolbook_block(0, 2, 2, 3, 0, 2)
    add_schoolbook_block(0, 2, 0, 2, 2, 3)
    add_schoolbook_block(0, 2, 2, 3, 2, 3)
    add_schoolbook_block(2, 3, 0, 2, 0, 2)
    add_schoolbook_block(2, 3, 2, 3, 0, 2)
    add_schoolbook_block(2, 3, 0, 2, 2, 3)
    add_schoolbook_block(2, 3, 2, 3, 2, 3)
    
    # Construct 3x3x2 part
    add_strassen_2x2x2(0, 0, 3)
    add_schoolbook_block(0, 2, 2, 3, 3, 5)
    add_schoolbook_block(2, 3, 0, 2, 3, 5)
    add_schoolbook_block(2, 3, 2, 3, 3, 5)
    
    if not U_cols:
        U = np.zeros((n*m, 1))
        V = np.zeros((m*p, 1))
        W = np.zeros((p*n, 1))
    else:
        U = np.array(U_cols, dtype=np.float64).T
        V = np.array(V_cols, dtype=np.float64).T
        W = np.array(W_cols, dtype=np.float64).T
        
    return {"U": U, "V": V, "W": W}
