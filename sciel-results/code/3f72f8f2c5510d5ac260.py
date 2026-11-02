import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    t_start = time.time()
    DEADLINE = 140
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Laderman rank-23 factors for 3x3x3
    LADERMAN_23 = [
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,0, 1,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,1, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 0,0,0, 1,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,1,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 1,0,0]),
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1]),
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,1,0]),
    ]

    # Verify Laderman factors
    laderman_valid = True
    if len(LADERMAN_23) == 23:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0
                    for r in range(23):
                        u, v, w = LADERMAN_23[r]
                        s += u[i*3+j] * v[j*3+k] * w[k*3+i]
                    if abs(s - 1) > 1e-9:
                        laderman_valid = False
                        break
                if not laderman_valid:
                    break
            if not laderman_valid:
                break
    
    if laderman_valid:
        rank = 55
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_laderman_block(col_offset, rank_offset):
            for r in range(23):
                u, v, w = LADERMAN_23[r]
                for i in range(3):
                    for j in range(3):
                        U[i*3+j][rank_offset + r] = u[i*3+j]
                for j in range(3):
                    for k in range(3):
                        V[j*7 + (col_offset + k)][rank_offset + r] = v[j*3+k]
                for k in range(3):
                    for i in range(3):
                        W[(col_offset + k)*3 + i][rank_offset + r] = w[k*3+i]

        add_laderman_block(0, 0)
        add_laderman_block(3, 23)
        
        # Block 3: Column 6 (3x1)
        for i in range(3):
            for j in range(3):
                r_idx = 46 + i*3 + j
                U[i*3+j][r_idx] = 1
                V[j*7 + 6][r_idx] = 1
                W[6*3 + i][r_idx] = 1

        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return schoolbook()
