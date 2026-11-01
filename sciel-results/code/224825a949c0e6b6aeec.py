import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit

    size_a = n * m
    size_b = m * p
    size_c = p * n

    # Build the true tensor T[a, b, c]
    T_true = np.zeros((size_a, size_b, size_c), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1

    # Strategy: Try multiple rank constructions, return the best valid one.
    # Best known constructions:
    # 1. Rank 28: For each k, two terms (one for j=0 covering both i, one for j=1 covering both i)
    # 2. Rank 49: Strassen on 2x2 blocks (7 per block * 7 blocks)
    # 3. Rank 56: Schoolbook

    best_result = None
    best_rank = n * m * p  # schoolbook as fallback

    # --- Construction 1: Rank 2p = 28 ---
    # For each k (0..13):
    #   Term k:    U[0]=1, U[2]=1, V[k]=1, W[2k]=1, W[2k+1]=1
    #   Term p+k:  U[1]=1, U[3]=1, V[p+k]=1, W[2k]=1, W[2k+1]=1
    rank_28 = 2 * p
    U_28 = np.zeros((size_a, rank_28), dtype=int)
    V_28 = np.zeros((size_b, rank_28), dtype=int)
    W_28 = np.zeros((size_c, rank_28), dtype=int)

    for k in range(p):
        # Term for j=0: covers (i=0,j=0,k) and (i=1,j=0,k)
        col = k
        U_28[0, col] = 1   # i=0, j=0
        U_28[2, col] = 1   # i=1, j=0
        V_28[0 * p + k, col] = 1  # j=0, k
        W_28[k * n + 0, col] = 1  # k, i=0
        W_28[k * n + 1, col] = 1  # k, i=1

        # Term for j=1: covers (i=0,j=1,k) and (i=1,j=1,k)
        col = p + k
        U_28[1, col] = 1   # i=0, j=1
        U_28[3, col] = 1   # i=1, j=1
        V_28[1 * p + k, col] = 1  # j=1, k
        W_28[k * n + 0, col] = 1  # k, i=0
        W_28[k * n + 1, col] = 1  # k, i=1

    # Verify rank 28
    T_recon_28 = np.einsum('ar,br,cr->abc', U_28, V_28, W_28, optimize=True)
    if np.array_equal(T_recon_28, T_true):
        best_rank = rank_28
        best_result = {"U": U_28.tolist(), "V": V_28.tolist(), "W": W_28.tolist()}
        # This is the best possible we know of; return immediately
        return best_result

    # --- Construction 2: Strassen rank 49 ---
    rank_49 = 7 * (p // 2)
    U_49 = np.zeros((size_a, rank_49), dtype=int)
    V_49 = np.zeros((size_b, rank_49), dtype=int)
    W_49 = np.zeros((size_c, rank_49), dtype=int)

    U_strassen = [
        [1, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0, 0, 0]
    ]
    V_strassen = [
        [1, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, -1, 0, 1],
        [1, 0, 0, 1, 0, -1, -1]
    ]
    W_strassen = [
        [1, -1, 1, 0, 0, 0, 0],
        [0, 1, 0, -1, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1]
    ]

    for b in range(p // 2):
        b_idx_map = [2 * b, 2 * b + 1, p + 2 * b, p + 2 * b + 1]
        c_idx_map = [4 * b, 4 * b + 2, 4 * b + 1, 4 * b + 3]

        for r in range(7):
            global_col = b * 7 + r
            for i_local in range(4):
                U_49[i_local, global_col] = U_strassen[i_local][r]
            for j_local in range(4):
                V_49[b_idx_map[j_local], global_col] = V_strassen[j_local][r]
            for k_local in range(4):
                W_49[c_idx_map[k_local], global_col] = W_strassen[k_local][r]

    T_recon_49 = np.einsum('ar,br,cr->abc', U_49, V_49, W_49, optimize=True)
    if np.array_equal(T_recon_49, T_true) and rank_49 < best_rank:
        best_rank = rank_49
        best_result = {"U": U_49.tolist(), "V": V_49.tolist(), "W": W_49.tolist()}

    # --- Fallback: Schoolbook rank 56 ---
    rank_school = n * m * p
    U_school = np.zeros((size_a, rank_school), dtype=int)
    V_school = np.zeros((size_b, rank_school), dtype=int)
    W_school = np.zeros((size_c, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    if best_result is None:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    return best_result
# EVOLVE-BLOCK-END
