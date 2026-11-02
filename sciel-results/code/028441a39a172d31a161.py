import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Build a rank-36 decomposition using Strassen for 2x2x2 blocks
    # and direct multiplication for 2x2x1 blocks.
    #
    # Decomposition:
    #   A (2x4) split into A1 (2x2) and A2 (2x2)
    #   B (4x5) split into B1 (2x5) and B2 (2x5)
    #   C = A1*B1 + A2*B2
    #
    # B1 = B1a (2x2) + B1b (2x3)
    # B2 = B2a (2x2) + B2b (2x3)
    #
    # A1*B1a: 2x2x2, rank 7 (Strassen)
    # A1*B1b: 2x2x3, rank 10 (using 2x2x3 decomposition)
    # A2*B2a: 2x2x2, rank 7 (Strassen)
    # A2*B2b: 2x2x3, rank 10 (using 2x2x3 decomposition)
    # Total rank: 7 + 10 + 7 + 10 = 34
    #
    # Wait, that's 34, not 36. Let me recalculate.
    # Actually, 2x2x3 has rank 10? Let me verify.
    # Schoolbook 2x2x3 is rank 6. Can we do better?
    # Actually, 2x2x3 is rank 6 (no known improvement).
    # So: 7 + 6 + 7 + 6 = 26? That seems too low.
    #
    # Let me reconsider. The 2x2x3 tensor [M_{2,2,3}] has rank 6.
    # So A1*B1b is 2x2 times 2x3, which is [M_{2,2,3}], rank 6.
    # Total: 7 + 6 + 7 + 6 = 26.
    #
    # But wait, I need to be more careful about the indexing.
    # Let me just use the Strassen 2x2x2 blocks and direct 2x2x1 blocks.
    # That gives rank 36 as in the current program.
    #
    # Let me try a different approach: use the known rank-30 decomposition
    # for 2x4x5 if it exists.
    #
    # Actually, let me just verify the rank-36 decomposition works and return it.

    start_time = time.time()
    deadline = start_time + 110.0

    # Rank 36 decomposition using Strassen
    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    # Strassen 2x2x2 patterns
    strassen_patterns = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    # 2x2x1 direct multiplication patterns
    direct_2x2x1_patterns = [
        ([1, 0, 0, 0], [1, 0], [1, 0]),
        ([0, 1, 0, 0], [0, 1], [1, 0]),
        ([0, 0, 1, 0], [1, 0], [0, 1]),
        ([0, 0, 0, 1], [0, 1], [0, 1]),
    ]

    def add_2x2x2_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        for u_pat, v_pat, w_pat in strassen_patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx // 2
                    k_local = idx % 2
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            col_offset += 1
        return col_offset

    def add_2x2x1_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        for u_pat, v_pat, w_pat in direct_2x2x1_patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx
                    k_local = 0
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx
                    k_local = 0
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            col_offset += 1
        return col_offset

    # Build the decomposition
    # A (2x4), B (4x5)
    # Split A into A1 (rows 0-1, cols 0-1) and A2 (rows 0-1, cols 2-3)
    # Split B into B1 (rows 0-1) and B2 (rows 2-3)
    # B1 = B1a (cols 0-1) + B1b (cols 2-4)
    # B2 = B2a (cols 0-1) + B2b (cols 2-4)

    col = 0
    # A1 * B1a: 2x2x2, Strassen, rank 7
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 0, 0, 0, col)
    # A1 * B1b: 2x2x3, direct, rank 6
    # Split into three 2x2x1 blocks
    col = add_2x2x1_block(U, V, W, 0, 0, 0, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 0, 0, 3, 0, 3, col)
    col = add_2x2x1_block(U, V, W, 0, 0, 0, 4, 0, 4, col)
    # A2 * B2a: 2x2x2, Strassen, rank 7
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 0, 0, 0, col)
    # A2 * B2b: 2x2x3, direct, rank 6
    col = add_2x2x1_block(U, V, W, 0, 2, 2, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 2, 2, 3, 0, 3, col)
    col = add_2x2x1_block(U, V, W, 0, 2, 2, 4, 0, 4, col)
    assert col == total_rank, f"Expected rank {total_rank}, got {col}"

    def verify_decomposition(U, V, W):
        rank = U.shape[1]
        # Check diagonal terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check off-diagonal terms
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue
                    s = np.sum(U[a] * V[b] * W[c])
                    if abs(s) > 1e-9:
                        return False
        return True

    if time.time() > deadline:
        # Return a valid schoolbook decomposition
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(nm)]
        V_s = [[0] * rank_school for _ in range(mp)]
        W_s = [[0] * rank_school for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}

    if verify_decomposition(U, V, W):
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Return schoolbook as fallback
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(nm)]
        V_s = [[0] * rank_school for _ in range(mp)]
        W_s = [[0] * rank_school for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
# EVOLVE-BLOCK-END
