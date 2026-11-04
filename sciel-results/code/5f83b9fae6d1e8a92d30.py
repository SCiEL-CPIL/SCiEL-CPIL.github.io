import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0

    # Schoolbook decomposition for fallback
    rank_schoolbook = n * m * p

    def schoolbook():
        rank = rank_schoolbook
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

    best_result = schoolbook()
    best_rank = rank_schoolbook

    # F(2) Strassen decomposition for 2x2x2 matrix multiplication
    F2_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    F2_V = [
        [1, 1, -1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, -1, 1, 0, 1],
    ]
    F2_W = [
        [1, 0, 1, -1, 0, 0, -1],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 1, -1, 0, 0, 0, 0],
        [1, -1, 0, 0, 1, 1, 0],
    ]

    # Strategy: split the 4x4x5 problem using a hybrid approach.
    # For the 4x4x4 part (k=0..3), use tensor product of F(2) to get rank 49.
    # For the 5th column (k=4), use schoolbook with rank 16.
    # Total: 49 + 16 = 65

    # Build 4x4x4x4 using tensor product of F(2) with itself
    U_44 = [[0] * 49 for _ in range(16)]
    V_44 = [[0] * 49 for _ in range(16)]
    W_44 = [[0] * 49 for _ in range(16)]

    for i1 in range(2):
        for i0 in range(2):
            for j1 in range(2):
                for j0 in range(2):
                    A_idx = i1*8 + i0*4 + j1*2 + j0
                    for r1 in range(7):
                        for r0 in range(7):
                            r = r1*7 + r0
                            U_44[A_idx][r] = F2_U[i1*2 + j1][r1] * F2_U[i0*2 + j0][r0]

    for j1 in range(2):
        for j0 in range(2):
            for k1 in range(2):
                for k0 in range(2):
                    B_idx = j1*8 + j0*4 + k1*2 + k0
                    for r1 in range(7):
                        for r0 in range(7):
                            r = r1*7 + r0
                            V_44[B_idx][r] = F2_V[j1*2 + k1][r1] * F2_V[j0*2 + k0][r0]

    for k1 in range(2):
        for k0 in range(2):
            for i1 in range(2):
                for i0 in range(2):
                    C_idx = k1*8 + k0*4 + i1*2 + i0
                    for r1 in range(7):
                        for r0 in range(7):
                            r = r1*7 + r0
                            W_44[C_idx][r] = F2_W[k1*2 + i1][r1] * F2_W[k0*2 + i0][r0]

    # Build full 4x4x4x5 decomposition
    total_rank = 49 + 16
    U_full = [[0] * total_rank for _ in range(16)]
    V_full = [[0] * total_rank for _ in range(20)]
    W_full = [[0] * total_rank for _ in range(20)]

    # Copy 4x4x4x4 part (k=0..3)
    for a_idx in range(16):
        for r in range(49):
            U_full[a_idx][r] = U_44[a_idx][r]

    for j in range(4):
        for k in range(4):
            b_idx = j*5 + k
            b_idx_44 = j*4 + k
            for r in range(49):
                V_full[b_idx][r] = V_44[b_idx_44][r]

    for k in range(4):
        for i in range(4):
            w_idx = k*4 + i
            w_idx_44 = k*4 + i
            for r in range(49):
                W_full[w_idx][r] = W_44[w_idx_44][r]

    # 5th column k=4: C[i,4] = sum_j A[i,j] * B[j,4]
    for i in range(4):
        for j in range(4):
            r = 49 + i*4 + j
            U_full[i*4+j][r] = 1
            V_full[j*5+4][r] = 1
            W_full[4*4+i][r] = 1

    # Verify full decomposition
    def check_decomposition(U, V, W):
        rank = len(U[0])
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(V[0]) != rank or len(W[0]) != rank:
            return False

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    if check_decomposition(U_full, V_full, W_full):
        best_result = {"U": U_full, "V": V_full, "W": W_full}
        best_rank = total_rank

    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_result

    return best_result
