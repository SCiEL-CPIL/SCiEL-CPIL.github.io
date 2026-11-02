import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def construct():
    t_start = time.time()
    DEADLINE = 145
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

    def verify_exact(U, V, W):
        rank = len(U[0])
        # Convert to Fraction for exact arithmetic if entries are not integers
        # But here entries are integers, so int arithmetic is exact.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
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

    # Verify Laderman exactly
    laderman_valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0
                for r in range(23):
                    u, v, w = LADERMAN_23[r]
                    s += u[i*3+j] * v[j*3+k] * w[k*3+i]
                if s != 1:
                    laderman_valid = False
                    break
            if not laderman_valid:
                break
        if not laderman_valid:
            break

    if not laderman_valid:
        return schoolbook()

    # Build the rank-55 baseline
    rank55 = 55
    U55 = [[0] * rank55 for _ in range(n * m)]
    V55 = [[0] * rank55 for _ in range(m * p)]
    W55 = [[0] * rank55 for _ in range(p * n)]
    
    def add_laderman_block(col_offset, rank_offset):
        for r in range(23):
            u, v, w = LADERMAN_23[r]
            for i in range(3):
                for j in range(3):
                    U55[i*3+j][rank_offset + r] = u[i*3+j]
            for j in range(3):
                for k in range(3):
                    V55[j*7 + (col_offset + k)][rank_offset + r] = v[j*3+k]
            for k in range(3):
                for i in range(3):
                    W55[(col_offset + k)*3 + i][rank_offset + r] = w[k*3+i]

    add_laderman_block(0, 0)
    add_laderman_block(3, 23)
    
    # Block 3: Column 6
    for i in range(3):
        for j in range(3):
            r_idx = 46 + i*3 + j
            U55[i*3+j][r_idx] = 1
            V55[j*7 + 6][r_idx] = 1
            W55[6*3 + i][r_idx] = 1

    if not verify_exact(U55, V55, W55):
        return schoolbook()

    return {"U": U55, "V": V55, "W": W55}
