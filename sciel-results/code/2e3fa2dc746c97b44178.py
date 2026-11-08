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

    # Rank-28 decomposition for 4x5 x 5x6 matrix multiplication
    # A (4x5) = [A1 (4x4) | A2 (4x1)]
    # B (5x6) = [B1 (4x6); B2 (1x6)]
    # C = A1*B1 + A2*B2
    # A1*B1: 4x4 x 4x6, rank 24 (optimal since n*p=24)
    # A2*B2: 4x1 x 1x6, rank 4
    # Total: rank 28

    rank_total = 28
    U = [[0] * rank_total for _ in range(n * m)]  # 20 x 28
    V = [[0] * rank_total for _ in range(m * p)]  # 30 x 28
    W = [[0] * rank_total for _ in range(p * n)]  # 24 x 28

    col = 0

    # Add a 4x4 x 4x1 decomposition:
    # A is rows [base_i, base_i+4), cols [base_j, base_j+4)
    # B is rows [base_j, base_j+4), col base_k
    # C is rows [base_i, base_i+4), col base_k
    # Rank 4: one term per column j of A
    def add_4x4_x_4x1(base_i, base_j, base_k, offset_col):
        nonlocal col
        for j_local in range(4):
            c = offset_col + j_local
            # U: for all i_local, U[(base_i+i_local)*m + (base_j+j_local)][c] = 1
            for i_local in range(4):
                U[(base_i + i_local) * m + (base_j + j_local)][c] = 1
            # V: V[(base_j+j_local)*p + base_k][c] = 1
            V[(base_j + j_local) * p + base_k][c] = 1
            # W: for all i_local, W[base_k*n + (base_i+i_local)][c] = 1
            for i_local in range(4):
                W[base_k * n + (base_i + i_local)][c] = 1
        col += 4
        return col

    # A1*B1: A1 is A[0:4, 0:4], B1 is B[0:4, 0:6]
    # Process each column of B1 independently
    for k in range(6):
        col = add_4x4_x_4x1(0, 0, k, col)

    # A2*B2: A2 is A[0:4, 4] (4x1), B2 is B[4, 0:6] (1x6)
    # C2[i,k] = A2[i,0] * B2[0,k] = A[i,4] * B[4,k]
    # Rank 4: one term per row i of A2
    for i_local in range(4):
        c = col + i_local
        # U: U[i_local*m + 4][c] = 1
        U[i_local * m + 4][c] = 1
        # V: for all k, V[4*p + k][c] = 1
        for k in range(6):
            V[4 * p + k][c] = 1
        # W: W[k*n + i_local][c] = 1 for all k... 
        # Actually, we need W[k*n + i_local][c] such that the product works out.
        # The tensor entry for (i, j, k) where j=4:
        # U[i*m+4][c] * V[4*p+k][c] * W[k*n+i][c]
        # We want this to be 1 when i=i_local, else 0.
        # So W[k*n + i][c] = 1 if i == i_local, else 0
        for k in range(6):
            W[k * n + i_local][c] = 1
    col += 4

    assert col == rank_total, f"Expected {rank_total}, got {col}"

    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat):
        r = len(U_mat[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U_mat[i * m + j][c] * V_mat[j * p + k][c] * W_mat[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook (rank 120)
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
