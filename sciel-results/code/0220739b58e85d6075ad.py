import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    rank = 85
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Helper to get 2x2 Strassen factors
    def get_strassen_2x2():
        u = np.zeros((4, 7))
        v = np.zeros((4, 7))
        w = np.zeros((4, 7))
        
        u[0, 0] = 1; u[3, 0] = 1
        v[0, 0] = 1; v[3, 0] = 1
        w[0, 0] = 1; w[3, 0] = 1
        
        u[2, 1] = 1; u[3, 1] = 1
        v[0, 1] = 1
        w[2, 1] = 1; w[3, 1] = -1
        
        u[0, 2] = 1
        v[1, 2] = 1; v[3, 2] = -1
        w[1, 2] = 1; w[3, 2] = 1
        
        u[3, 3] = 1
        v[2, 3] = 1; v[0, 3] = -1
        w[0, 3] = 1; w[2, 3] = 1
        
        u[0, 4] = 1; u[1, 4] = 1
        v[3, 4] = 1
        w[0, 4] = -1; w[1, 4] = 1
        
        u[0, 5] = -1; u[2, 5] = 1
        v[0, 5] = 1; v[1, 5] = 1
        w[3, 5] = 1
        
        u[1, 6] = 1; u[3, 6] = -1
        v[2, 6] = 1; v[3, 6] = 1
        w[0, 6] = 1
        
        return u, v, w

    s_u, s_v, s_w = get_strassen_2x2()

    # Define the 7 terms for 4x4x4 Strassen
    # Each term is a tuple of (a_blocks, b_blocks, c_blocks)
    # blocks are dicts mapping (row, col) to coeff
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    col = 0
    for t_idx in range(7):
        a_c, b_c, c_c = terms[t_idx]
        for k in range(7):
            c_idx = col
            # Apply 2x2 Strassen factor k to the blocks
            for (br, bc), coeff in a_c.items():
                for r in range(2):
                    for c in range(2):
                        idx = r * 2 + c
                        if s_u[idx, k] != 0:
                            gr = 2 * br + r
                            gc = 2 * bc + c
                            U[gr * 5 + gc, c_idx] += coeff * s_u[idx, k]
            
            for (br, bc), coeff in b_c.items():
                for r in range(2):
                    for c in range(2):
                        idx = r * 2 + c
                        if s_v[idx, k] != 0:
                            gr = 2 * br + r
                            gc = 2 * bc + c
                            V[gr * 5 + gc, c_idx] += coeff * s_v[idx, k]
            
            for (br, bc), coeff in c_c.items():
                for r in range(2):
                    for c in range(2):
                        idx = r * 2 + c
                        if s_w[idx, k] != 0:
                            gr = 2 * br + r
                            gc = 2 * bc + c
                            W[gc * 4 + gr, c_idx] += coeff * s_w[idx, k]
            
            col += 1

    # Add remaining terms for 4x4x1, 4x1x4, 4x1x1
    # 4x4x1: A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1
            col += 1

    # 4x1x4: A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4, col] = 1
            V[4 * 5 + k, col] = 1
            W[k * 4 + i, col] = 1
            col += 1

    # 4x1x1: A[0:4, 4] * B[4, 4] -> C[0:4, 4]
    for i in range(4):
        U[i * 5 + 4, col] = 1
        V[4 * 5 + 4, col] = 1
        W[4 * 4 + i, col] = 1
        col += 1

    if time.time() > deadline:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
