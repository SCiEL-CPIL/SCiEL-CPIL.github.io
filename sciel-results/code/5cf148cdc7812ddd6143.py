import time
import sys

def construct():
    n, m, p = 2, 4, 5
    nm, mp, pn = n * m, m * p, p * n  # 8, 20, 10

    # Strassen 2x2x2 patterns: (u_pattern, v_pattern, w_pattern)
    # u_pattern[i*2+j] corresponds to A[i,j]
    # v_pattern[j*2+k] corresponds to B[j,k]
    # w_pattern[k*2+i] corresponds to C[k,i] (trace convention W[k*n+i])

    strassen_patterns = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    # Schoolbook 2x2x1 patterns
    schoolbook_2x2x1 = [
        ([1, 0, 0, 0], [1, 0], [1, 0]),
        ([0, 1, 0, 0], [0, 1], [1, 0]),
        ([0, 0, 1, 0], [1, 0], [0, 1]),
        ([0, 0, 0, 1], [0, 1], [0, 1]),
    ]

    rank_total = 36
    U = [[0] * rank_total for _ in range(nm)]
    V = [[0] * rank_total for _ in range(mp)]
    W = [[0] * rank_total for _ in range(pn)]

    def add_strassen_block(u_i_off, u_j_off, v_j_off, v_k_off, w_i_off, w_k_off, col_off):
        for u_pat, v_pat, w_pat in strassen_patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = u_i_off + i_local
                    j_global = u_j_off + j_local
                    U[i_global * m + j_global][col_off] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    j_global = v_j_off + j_local
                    k_global = v_k_off + k_local
                    V[j_global * p + k_global][col_off] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx // 2
                    k_local = idx % 2
                    i_global = w_i_off + i_local
                    k_global = w_k_off + k_local
                    W[k_global * n + i_global][col_off] = val
            col_off += 1
        return col_off

    def add_schoolbook_2x2x1(u_i_off, u_j_off, v_j_off, v_k_off, w_i_off, w_k_off, col_off):
        for u_pat, v_pat, w_pat in schoolbook_2x2x1:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = u_i_off + i_local
                    j_global = u_j_off + j_local
                    U[i_global * m + j_global][col_off] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx
                    k_local = 0
                    j_global = v_j_off + j_local
                    k_global = v_k_off + k_local
                    V[j_global * p + k_global][col_off] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx
                    k_local = 0
                    i_global = w_i_off + i_local
                    k_global = w_k_off + k_local
                    W[k_global * n + i_global][col_off] = val
            col_off += 1
        return col_off

    col = 0
    col = add_strassen_block(0, 0, 0, 0, 0, 0, col)
    col = add_strassen_block(0, 2, 2, 0, 0, 0, col)
    col = add_strassen_block(0, 0, 0, 2, 0, 2, col)
    col = add_strassen_block(0, 2, 2, 2, 0, 2, col)
    col = add_schoolbook_2x2x1(0, 0, 0, 4, 0, 4, col)
    col = add_schoolbook_2x2x1(0, 2, 2, 4, 0, 4, col)

    assert col == rank_total, f"Expected {rank_total}, got {col}"

    def verify():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank_total):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
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
                    s = 0
                    for r in range(rank_total):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s) > 1e-9:
                        return False
        return True

    if verify():
        return {"U": U, "V": V, "W": W}
    else:
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(nm)]
        V_sb = [[0] * rank_sb for _ in range(mp)]
        W_sb = [[0] * rank_sb for _ in range(pn)]
        col2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col2] = 1
                    V_sb[j * p + k][col2] = 1
                    W_sb[k * n + i][col2] = 1
                    col2 += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
