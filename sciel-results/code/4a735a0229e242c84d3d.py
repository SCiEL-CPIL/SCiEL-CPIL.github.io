import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    t0 = time.time()
    deadline = 140.0
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p

    # Fallback
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Try Rank 34 (2x2x3 + 2x2x2 decomposition)
    total_rank = 34
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    def add_2x2x2_block(U, V, W, a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, col_off):
        # Standard Strassen for 2x2x2
        patterns = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_l = idx // 2; j_l = idx % 2
                    U[(a_i_off + i_l) * m + (a_j_off + j_l), col_off] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_l = idx // 2; k_l = idx % 2
                    V[(b_j_off + j_l) * p + (b_k_off + k_l), col_off] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_l = idx // 2; k_l = idx % 2
                    W[(c_k_off + k_l) * n + (c_i_off + i_l), col_off] = val
            col_off += 1
        return col_off

    def add_2x2x3_block(U, V, W, a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, col_off):
        # Rank 10 decomposition for 2x2x3
        # First 7 terms: Strassen for first 2 columns (k=0,1)
        col_off = add_2x2x2_block(U, V, W, a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, col_off)
        
        # Next 3 terms for the 3rd column (k=2)
        # C_i2 = A_i0 * B_02 + A_i1 * B_12
        # We need to add 3 more rank-1 terms to cover this
        # Term 8: A_00 * B_02
        U[(a_i_off + 0) * m + (a_j_off + 0), col_off] = 1
        V[(b_j_off + 0) * p + (b_k_off + 2), col_off] = 1
        W[(c_k_off + 2) * n + (c_i_off + 0), col_off] = 1
        col_off += 1
        
        # Term 9: A_10 * B_02
        U[(a_i_off + 1) * m + (a_j_off + 0), col_off] = 1
        V[(b_j_off + 0) * p + (b_k_off + 2), col_off] = 1
        W[(c_k_off + 2) * n + (c_i_off + 1), col_off] = 1
        col_off += 1
        
        # Term 10: A_01 * B_12
        U[(a_i_off + 0) * m + (a_j_off + 1), col_off] = 1
        V[(b_j_off + 1) * p + (b_k_off + 2), col_off] = 1
        W[(c_k_off + 2) * n + (c_i_off + 0), col_off] = 1
        col_off += 1
        
        # Term 11: A_11 * B_12 - but wait, we only have 10 terms total
        # Let me reconsider. The 2x2x3 tensor has 12 entries, so we need at least 12 rank-1 terms?
        # No, the bilinear rank can be less. Let me check if 10 is actually achievable.
        
        # Actually, I think I made an error. Let me use a verified 10-term decomposition.
        # For now, let's fall back to the 36 solution if verification fails.
        return col_off

    # Since the 2x2x3 rank-10 decomposition is non-trivial, let's use the known 36 solution
    # but with a cleaner implementation
    
    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    def add_2x2x1_block(U, V, W, a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, col_off):
        patterns = [
            ([1, 0, 0, 0], [1, 0], [1, 0]),
            ([0, 1, 0, 0], [0, 1], [1, 0]),
            ([0, 0, 1, 0], [1, 0], [0, 1]),
            ([0, 0, 0, 1], [0, 1], [0, 1]),
        ]
        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_l = idx // 2; j_l = idx % 2
                    U[(a_i_off + i_l) * m + (a_j_off + j_l), col_off] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_l = idx; k_l = 0
                    V[(b_j_off + j_l) * p + (b_k_off + k_l), col_off] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_l = idx; k_l = 0
                    W[(c_k_off + k_l) * n + (c_i_off + i_l), col_off] = val
            col_off += 1
        return col_off

    col = 0
    # Block 1: A(0,1) x B(0,1) -> C(0,1)
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 0, 0, 0, col)
    # Block 2: A(0,1) x B(2,3) -> C(0,2)
    col = add_2x2x2_block(U, V, W, 0, 0, 2, 2, 0, 2, col)
    # Block 3: A(0,1) x B(4) -> C(0,4)
    col = add_2x2x1_block(U, V, W, 0, 0, 4, 4, 0, 4, col)
    # Block 4: A(2,3) x B(0,1) -> C(2,0)
    col = add_2x2x2_block(U, V, W, 0, 2, 0, 0, 2, 0, col)
    # Block 5: A(2,3) x B(2,3) -> C(2,2)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 2, 2, 2, col)
    # Block 6: A(2,3) x B(4) -> C(2,4)
    col = add_2x2x1_block(U, V, W, 0, 2, 4, 4, 2, 4, col)

    assert col == total_rank

    # Verify
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        for a in range(nm):
            for b in range(mp):
                if a % m != b // p: continue
                for c in range(pn):
                    if a % m == b // p and b % p == c // n and a // m == c % n: continue
                    s = np.sum(U[a] * V[b] * W[c])
                    if abs(s) > 1e-9:
                        return False
        return True

    if verify(U, V, W):
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return best_result
