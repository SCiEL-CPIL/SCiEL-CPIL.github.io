"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    import time
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # 2x2x2 Strassen rank-7 decomposition
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],   # (0,0)
        [0, 0, 0, 0, 1, 0, 1],    # (0,1)
        [0, 1, 0, 0, 0, 1, 0],    # (1,0)
        [1, 1, 0, 1, 0, 0, -1],   # (1,1)
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],   # (0,0)
        [0, 0, 1, 0, 0, 1, 0],    # (0,1)
        [0, 0, 0, 1, 0, 0, 1],    # (1,0)
        [1, 0, -1, 0, 1, 0, 1],   # (1,1)
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],   # (0,0)
        [0, 1, 0, 1, 0, 0, 0],    # (0,1)
        [0, 0, 1, 0, 1, 0, 0],    # (1,0)
        [1, -1, 1, 0, 0, 1, 0],   # (1,1)
    ]

    # Build 4x4x4 Strassen decomposition (rank 49)
    # The 4x4x4 tensor is a 2x2x2 tensor of 2x2x2 tensors.
    # We use the 2x2x2 Strassen decomposition at the block level.
    # Each block product is decomposed using the 2x2x2 Strassen decomposition.
    # Total rank: 7 (block level) * 7 (sub-block level) = 49

    # For 4x4x4, the factors are:
    # U: 16 x 49 (i*4+j for i,j in 0..3, r in 0..48)
    # V: 16 x 49 (j*4+k for j,k in 0..3, r in 0..48)
    # W: 16 x 49 (k*4+i for k,i in 0..3, r in 0..48)

    # The 49 terms are indexed by (block_term, sub_term) where block_term in 0..6, sub_term in 0..6
    # U[i*4+j, block_term*7+sub_term] = S_U[i_block*2+i_sub, block_term] * S_U[j_block*2+j_sub, sub_term]
    # Wait, that's not quite right. Let me think more carefully.

    # Actually, for the 4x4x4 case, we split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x2

    # Strassen at block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A11)(B01+B11)

    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7

    # Each M_k is a 2x2 matrix product, which we decompose using Strassen (rank 7).
    # So M_k = sum_{s=0}^{6} u_s * v_s * w_s where u_s, v_s, w_s are 2x2 matrices (flattened to vectors of length 4).

    # For the 4x4x4 decomposition, each of the 49 terms corresponds to a specific (k, s) pair.
    # The U factor for term (k,s) and position (i,j):
    #   U[i*4+j, k*7+s] = U_block[k, i_block] * U_sub[s, i_sub*2+j_sub]
    # where i = i_block*2 + i_sub, j = j_block*2 + j_sub

    # Let me define the block-level factors more carefully.
    # For the block-level Strassen, we have 7 terms.
    # For each term k, the U factor at block position (i_block, j_block) is:
    #   U_block[k, i_block*2+j_block] (flattened 2x2)

    # Let me just build this programmatically.

    # Block-level Strassen U factors (for the 2x2x2 block structure)
    # These are 4x7 matrices where row index is i_block*2+j_block
    U_block = [
        [1, 0, 1, 0, 1, -1, 0],   # (0,0)
        [0, 0, 0, 0, 1, 0, 1],    # (0,1)
        [0, 1, 0, 0, 0, 1, 0],    # (1,0)
        [1, 1, 0, 1, 0, 0, -1],   # (1,1)
    ]
    V_block = [
        [1, 1, 0, -1, 0, 1, 0],   # (0,0)
        [0, 0, 1, 0, 0, 1, 0],    # (0,1)
        [0, 0, 0, 1, 0, 0, 1],    # (1,0)
        [1, 0, -1, 0, 1, 0, 1],   # (1,1)
    ]
    W_block = [
        [1, 0, 0, 1, -1, 0, 1],   # (0,0)
        [0, 1, 0, 1, 0, 0, 0],    # (0,1)
        [0, 0, 1, 0, 1, 0, 0],    # (1,0)
        [1, -1, 1, 0, 0, 1, 0],   # (1,1)
    ]

    # Sub-block level Strassen factors (same as S_U, S_V, S_W)
    U_sub = S_U
    V_sub = S_V
    W_sub = S_W

    # Build 4x4x4 factors
    U_444 = [[0] * 49 for _ in range(16)]
    V_444 = [[0] * 49 for _ in range(16)]
    W_444 = [[0] * 49 for _ in range(16)]

    for k in range(7):  # block-level term
        for s in range(7):  # sub-block-level term
            col = k * 7 + s
            for ib in range(2):
                for jb in range(2):
                    i = ib * 2
                    j = jb * 2
                    # U factor
                    U_444[i * 4 + j][col] = U_block[ib * 2 + jb][k] * U_sub[ib * 2 + jb][s]
                    # Actually, this isn't quite right either. Let me reconsider.
                    # The U factor for position (i,j) in the 4x4x4 case should be:
                    # U[i*4+j, k*7+s] = U_block[k, ib*2+jb] * U_sub[s, ib*2+jb]
                    # But U_sub[s, ib*2+jb] doesn't make sense because U_sub is indexed by (sub_i, sub_j).

    # Let me restart this more carefully.
    # For the 4x4x4 case, we have positions (i, j, k) where i, j, k in 0..3.
    # We split i = ib*2 + ii, j = jb*2 + jj, k = kb*2 + kk where ib, jb, kb, ii, jj, kk in 0..1.

    # The tensor is T[i,j,k] = sum_{r=0}^{48} U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]

    # Each term r corresponds to (block_term, sub_term) = (bt, st) where r = bt*7 + st.

    # U[i*4+j, bt*7+st] = U_block[bt, ib*2+jj] * U_sub[st, ii*2+jj]
    # Wait, no. Let me think about what the factors should be.

    # For the block-level decomposition, the U factor at block position (ib, jb) for term bt is U_block[bt, ib*2+jb].
    # For the sub-block level, within each block, the U factor at sub-position (ii, jj) for term st is U_sub[st, ii*2+jj].

    # So the full U factor is: U[i*4+j, bt*7+st] = U_block[bt, ib*2+jb] * U_sub[st, ii*2+jj]
    # where i = ib*2+ii, j = jb*2+jj.

    # Similarly:
    # V[j*4+k, bt*7+st] = V_block[bt, jb*2+kk] * V_sub[st, jj*2+kk]
    # W[k*4+i, bt*7+st] = W_block[bt, kb*2+ii] * W_sub[st, kk*2+ii]

    U_444 = [[0] * 49 for _ in range(16)]
    V_444 = [[0] * 49 for _ in range(16)]
    W_444 = [[0] * 49 for _ in range(16)]

    for bt in range(7):
        for st in range(7):
            col = bt * 7 + st
            for ib in range(2):
                for ii in range(2):
                    i = ib * 2 + ii
                    for jb in range(2):
                        for jj in range(2):
                            j = jb * 2 + jj
                            U_444[i * 4 + j][col] = U_block[ib * 2 + jb][bt] * U_sub[ii * 2 + jj][st]
            for jb in range(2):
                for jj in range(2):
                    j = jb * 2 + jj
                    for kb in range(2):
                        for kk in range(2):
                            k = kb * 2 + kk
                            V_444[j * 4 + k][col] = V_block[jb * 2 + kk][bt] * V_sub[jj * 2 + kk][st]
            for kb in range(2):
                for kk in range(2):
                    k = kb * 2 + kk
                    for ib in range(2):
                        for ii in range(2):
                            i = ib * 2 + ii
                            W_444[k * 4 + i][col] = W_block[kb * 2 + ii][bt] * W_sub[kk * 2 + ii][st]

    # Now build the 4x4x2 part (rank 28 using block-wise Strassen)
    # This is for k = 0, 1 (or k = 2, 3, 4, 5 depending on how we split)
    # Actually, let's split the 4x4x6 into 4x4x4 (k=0..3) and 4x4x2 (k=4,5)
    # For the 4x4x2 part, we use block-wise Strassen with 2x2x2 blocks.
    # There are 2 (i blocks) * 2 (j blocks) * 1 (k block of size 2) = 4 blocks.
    # Each block has rank 7, so total rank = 4 * 7 = 28.

    U_442 = [[0] * 28 for _ in range(16)]
    V_442 = [[0] * 28 for _ in range(8)]  # j*2+k for j in 0..3, k in 0..1
    W_442 = [[0] * 28 for _ in range(8)]  # k*4+i for k in 0..1, i in 0..3

    col_offset = 0
    for bi in range(2):  # i block
        for bj in range(2):  # j block
            for bk in range(1):  # k block (size 2)
                for c in range(7):
                    col = col_offset + c
                    for di in range(2):
                        for dj in range(2):
                            i = bi * 2 + di
                            j = bj * 2 + dj
                            U_442[i * 4 + j][col] = S_U[di * 2 + dj][c]
                    for dj in range(2):
                        for dk in range(2):
                            j = bj * 2 + dj
                            k = bk * 2 + dk
                            V_442[j * 2 + k][col] = S_V[dj * 2 + dk][c]
                    for dk in range(2):
                        for di in range(2):
                            k = bk * 2 + dk
                            i = bi * 2 + di
                            W_442[k * 4 + i][col] = S_W[dk * 2 + di][c]
                col_offset += 7

    # Now combine the two parts
    # Total rank = 49 + 28 = 77
    total_rank = 77

    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(24)]  # j*6+k for j in 0..3, k in 0..5
    W = [[0] * total_rank for _ in range(24)]  # k*4+i for k in 0..5, i in 0..3

    # Copy 4x4x4 part (k=0..3)
    for r in range(49):
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][r] = U_444[i * 4 + j][r]
        for j in range(4):
            for k in range(4):
                V[j * 6 + k][r] = V_444[j * 4 + k][r]
        for k in range(4):
            for i in range(4):
                W[k * 4 + i][r] = W_444[k * 4 + i][r]

    # Copy 4x4x2 part (k=4,5)
    for r in range(28):
        col = 49 + r
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][col] = U_442[i * 4 + j][r]
        for j in range(4):
            for k in range(2):
                V[j * 6 + (4 + k)][col] = V_442[j * 2 + k][r]
        for k in range(2):
            for i in range(4):
                W[(4 + k) * 4 + i][col] = W_442[k * 4 + i][r]

    # Verify the decomposition
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    def compute_tensor(U_arr, V_arr, W_arr, n, m, p):
        rank = U_arr.shape[1]
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U_arr[:, r]
            v_col = V_arr[:, r]
            w_col = W_arr[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
        return T

    T_check = compute_tensor(U_np, V_np, W_np, n, m, p)
    if not np.allclose(T_check, 1.0, atol=1e-9):
        # Fall back to block-wise Strassen if the new decomposition is invalid
        n_blocks = n // 2
        m_blocks = m // 2
        p_blocks = p // 2
        total_rank_fb = n_blocks * m_blocks * p_blocks * 7
        U_fb = [[0] * total_rank_fb for _ in range(n * m)]
        V_fb = [[0] * total_rank_fb for _ in range(m * p)]
        W_fb = [[0] * total_rank_fb for _ in range(p * n)]
        col_offset = 0
        for bi in range(n_blocks):
            for bj in range(m_blocks):
                for bk in range(p_blocks):
                    for c in range(7):
                        col = col_offset + c
                        for di in range(2):
                            for dj in range(2):
                                i = bi * 2 + di
                                j = bj * 2 + dj
                                U_fb[i * m + j][col] = S_U[di * 2 + dj][c]
                        for dj in range(2):
                            for dk in range(2):
                                j = bj * 2 + dj
                                k = bk * 2 + dk
                                V_fb[j * p + k][col] = S_V[dj * 2 + dk][c]
                        for dk in range(2):
                            for di in range(2):
                                k = bk * 2 + dk
                                i = bi * 2 + di
                                W_fb[k * n + i][col] = S_W[dk * 2 + di][c]
                    col_offset += 7
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    # Try to remove columns
    removed = set()
    for r in range(total_rank):
        if time.time() > deadline - 10:
            break
        if r in removed:
            continue
        U_trial = U_np.copy()
        V_trial = V_np.copy()
        W_trial = W_np.copy()
        U_trial[:, r] = 0
        V_trial[:, r] = 0
        W_trial[:, r] = 0
        T_trial = compute_tensor(U_trial, V_trial, W_trial, n, m, p)
        if np.allclose(T_trial, 1.0, atol=1e-9):
            removed.add(r)
            U_np = U_trial
            V_np = V_trial
            W_np = W_trial

    if removed:
        keep_cols = [c for c in range(total_rank) if c not in removed]
        U_final = U_np[:, keep_cols]
        V_final = V_np[:, keep_cols]
        W_final = W_np[:, keep_cols]
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
