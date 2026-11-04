import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 vectors
    # u, v, w are length 4
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    S_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    S_v = [
        [1, 1, 0, 0], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    S_w = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 0, 1, -1], # M2 -> C10, -C11
        [0, 1, 0, 1], # M3 -> C01, C11
        [1, 0, 1, 0], # M4 -> C00, C10
        [-1, 1, 0, 0], # M5 -> -C00, C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ]
    
    # Standard 2x2x1 vectors
    # u, v, w are length 4, 2, 2
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(1,0)
    # C: 0:(0,0), 1:(1,0)
    D_u = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    D_v = [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ]
    D_w = [
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1]
    ]
    
    def add_strassen(u_row_map, v_row_map, w_row_map):
        nonlocal col
        for r in range(7):
            for local_idx, val in enumerate(S_u[r]):
                if val != 0:
                    U[u_row_map[local_idx]][col] += val
            for local_idx, val in enumerate(S_v[r]):
                if val != 0:
                    V[v_row_map[local_idx]][col] += val
            for local_idx, val in enumerate(S_w[r]):
                if val != 0:
                    W[w_row_map[local_idx]][col] += val
            col += 1

    def add_standard(u_row_map, v_row_map, w_row_map):
        nonlocal col
        for r in range(4):
            for local_idx, val in enumerate(D_u[r]):
                if val != 0:
                    U[u_row_map[local_idx]][col] += val
            for local_idx, val in enumerate(D_v[r]):
                if val != 0:
                    V[v_row_map[local_idx]][col] += val
            for local_idx, val in enumerate(D_w[r]):
                if val != 0:
                    W[w_row_map[local_idx]][col] += val
            col += 1
            
    # Block 1: A1 (rows 0,1; cols 0,1) * B1a (rows 0,1; cols 0,1)
    # u_map: local (i, j) -> global (i, j)
    # u_row_map[0] = 0*4+0=0, [1]=1, [2]=4, [3]=5
    u_map_1 = [0, 1, 4, 5]
    # v_map: local (j, k) -> global (j, k)
    # v_row_map[0] = 0*5+0=0, [1]=1, [2]=5, [3]=6
    v_map_1 = [0, 1, 5, 6]
    # w_map: local (i, k) -> global (i, k)
    # w_row_map[0] = 0*2+0=0, [1]=1, [2]=2, [3]=3
    w_map_1 = [0, 1, 2, 3]
    add_strassen(u_map_1, v_map_1, w_map_1)
    
    # Block 2: A1 * B1b (rows 0,1; cols 2,3)
    u_map_2 = [0, 1, 4, 5]
    # v_map: local (j, k) -> global (j, k+2)
    # v_row_map[0] = 0*5+2=2, [1]=3, [2]=7, [3]=8
    v_map_2 = [2, 3, 7, 8]
    # w_map: local (i, k) -> global (i, k+2)
    # w_row_map[0] = 2*2+0=4, [1]=5, [2]=6, [3]=7
    w_map_2 = [4, 5, 6, 7]
    add_strassen(u_map_2, v_map_2, w_map_2)
    
    # Block 3: A1 * B1c (rows 0,1; col 4)
    u_map_3 = [0, 1, 4, 5]
    # v_map: local (j, 0) -> global (j, 4)
    # v_row_map[0] = 0*5+4=4, [1]=9
    v_map_3 = [4, 9]
    # w_map: local (i, 0) -> global (i, 4)
    # w_row_map[0] = 4*2+0=8, [1]=9
    w_map_3 = [8, 9]
    add_standard(u_map_3, v_map_3, w_map_3)
    
    # Block 4: A2 (rows 0,1; cols 2,3) * B2a (rows 2,3; cols 0,1)
    # u_map: local (i, j) -> global (i, j+2)
    # u_row_map[0] = 0*4+2=2, [1]=3, [2]=6, [3]=7
    u_map_4 = [2, 3, 6, 7]
    # v_map: local (j, k) -> global (j+2, k)
    # v_row_map[0] = 2*5+0=10, [1]=11, [2]=15, [3]=16
    v_map_4 = [10, 11, 15, 16]
    # w_map: local (i, k) -> global (i, k)
    w_map_4 = [0, 1, 2, 3]
    add_strassen(u_map_4, v_map_4, w_map_4)
    
    # Block 5: A2 * B2b (rows 2,3; cols 2,3)
    u_map_5 = [2, 3, 6, 7]
    # v_map: local (j, k) -> global (j+2, k+2)
    # v_row_map[0] = 2*5+2=12, [1]=13, [2]=17, [3]=18
    v_map_5 = [12, 13, 17, 18]
    # w_map: local (i, k) -> global (i, k+2)
    w_map_5 = [4, 5, 6, 7]
    add_strassen(u_map_5, v_map_5, w_map_5)
    
    # Block 6: A2 * B2c (rows 2,3; col 4)
    u_map_6 = [2, 3, 6, 7]
    # v_map: local (j, 0) -> global (j+2, 4)
    # v_row_map[0] = 2*5+4=14, [1]=19
    v_map_6 = [14, 19]
    # w_map: local (i, 0) -> global (i, 4)
    w_map_6 = [8, 9]
    add_standard(u_map_6, v_map_6, w_map_6)
    
    return {"U": U, "V": V, "W": W}
