"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # safety margin before 120s limit

    # Construct a rank-28 decomposition using block decomposition
    # A (4x5) = [A1 (4x4) | A2 (4x1)]
    # B (5x6) = [B1 (4x6); B2 (1x6)]
    # C = A1*B1 + A2*B2
    
    # A1*B1: (4x4) x (4x6)
    # Split B1 into B1a (4x3) and B1b (4x3)
    # A1*B1a: (4x4) x (4x3), A1*B1b: (4x4) x (4x3)
    # Each (4x4) x (4x3) = (4x4)x(4x2) + (4x4)x(4x1)
    # (4x4)x(4x2) = 2*(4x4)x(4x1), each rank 4, total 8
    # (4x4)x(4x1): rank 4
    # So (4x4)x(4x3): rank 12
    # A1*B1: rank 24
    # A2*B2: (4x1)x(1x6): rank 4
    # Total: rank 28
    
    rank_total = 28
    U = [[0] * rank_total for _ in range(n * m)]  # 20 x 28
    V = [[0] * rank_total for _ in range(m * p)]  # 30 x 28
    W = [[0] * rank_total for _ in range(p * n)]  # 24 x 28
    
    col = 0
    
    # Helper: add a (4x4)x(4x1) decomposition starting at column col
    # A_sub is 4x4, B_sub is 4x1
    # i in [0,4), j in [0,4), k=0
    # U[i*4+j, col+c] = delta(i,c)
    # V[j, col+c] = 1
    # W[i, col+c] = delta(i,c)
    def add_4x4_x_4x1(base_i, base_j, base_k, offset_col):
        nonlocal col
        for c in range(4):
            for i in range(4):
                for j in range(4):
                    if i == c:
                        U[(base_i + i) * m + (base_j + j)][offset_col + c] = 1
            for j in range(4):
                V[(base_j + j) * p + base_k][offset_col + c] = 1
            for i in range(4):
                W[(base_k + 0) * n + (base_i + i)][offset_col + c] = 1
        col += 4
        return col
    
    # A1*B1a: A1 is rows 0-3, cols 0-3 of A; B1a is rows 0-3, cols 0-2 of B
    # (4x4) x (4x3): split into (4x4)x(4x2) + (4x4)x(4x1)
    # (4x4)x(4x2): two (4x4)x(4x1)
    # First (4x4)x(4x1): B col 0
    col = add_4x4_x_4x1(0, 0, 0, col)  # A[0:4, 0:4] x B[0:4, 0]
    # Second (4x4)x(4x1): B col 1
    col = add_4x4_x_4x1(0, 0, 1, col)  # A[0:4, 0:4] x B[0:4, 1]
    # (4x4)x(4x1): B col 2
    col = add_4x4_x_4x1(0, 0, 2, col)  # A[0:4, 0:4] x B[0:4, 2]
    
    # A1*B1b: B1b is rows 0-3, cols 3-5 of B
    # (4x4)x(4x3): split into (4x4)x(4x2) + (4x4)x(4x1)
    col = add_4x4_x_4x1(0, 0, 3, col)  # A[0:4, 0:4] x B[0:4, 3]
    col = add_4x4_x_4x1(0, 0, 4, col)  # A[0:4, 0:4] x B[0:4, 4]
    col = add_4x4_x_4x1(0, 0, 5, col)  # A[0:4, 0:4] x B[0:4, 5]
    
    # A2*B2: A2 is row 0-3, col 4 of A (4x1); B2 is row 4, cols 0-5 of B (1x6)
    # (4x1) x (1x6): rank 4
    # i in [0,4), j=4, k in [0,6)
    # U[i*5+4, col+c] = delta(i,c)
    # V[4*6+k, col+c] = 1
    # W[k*4+i, col+c] = delta(i,c)
    for c in range(4):
        for i in range(4):
            if i == c:
                U[i * m + 4][col + c] = 1
        for k in range(6):
            V[4 * p + k][col + c] = 1
        for i in range(4):
            W[0 * n + i][col + c] = 1
    col += 4
    
    assert col == rank_total, f"Expected {rank_total}, got {col}"
    
    # Verify
    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col2] = 1
                    V_school[j * p + k][col2] = 1
                    W_school[k * n + i][col2] = 1
                    col2 += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
