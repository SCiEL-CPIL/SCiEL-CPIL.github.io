import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 130  # seconds, safety margin before 150s limit

    # Rank 65: 49 (Strassen for 4x4x4) + 16 (Schoolbook for 4x4x1)
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65

    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0

    # Part 1: Strassen 4x4x4 (rank 49)
    # Strassen 2x2x2 decomposition
    inner_defs = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,0,1),(1,1,1)]),       # M1
        ([(2,1),(3,1)], [(0,1)], [(1,0,1),(1,1,-1)]),             # M2
        ([(0,1)], [(1,1),(3,-1)], [(0,1,1),(1,1,1)]),             # M3
        ([(3,1)], [(2,1),(0,-1)], [(0,0,1),(1,0,1)]),             # M4
        ([(0,1),(1,1)], [(3,1)], [(0,0,-1),(0,1,1)]),             # M5
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(1,1,1)]),               # M6
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,0,1)]),               # M7
    ]

    super_defs = [
        ([1,0,0,1], [1,0,0,1], [(0,0,1),(1,1,1)]),       # M1 -> C00, C11
        ([0,0,1,1], [1,0,0,0], [(1,0,1),(1,1,-1)]),      # M2 -> C10, -C11
        ([1,0,0,0], [0,1,0,-1], [(0,1,1),(1,1,1)]),      # M3 -> C01, C11
        ([0,0,0,1], [-1,0,1,0], [(0,0,1),(1,0,1)]),      # M4 -> C00, C10
        ([1,1,0,0], [0,0,0,1], [(0,0,-1),(0,1,1)]),      # M5 -> -C00, C01
        ([-1,0,1,0], [1,1,0,0], [(1,1,1)]),              # M6 -> C11
        ([0,1,0,-1], [0,0,1,1], [(0,0,1)]),              # M7 -> C00
    ]

    def get_p_element(p_elem_idx, p_block_coeffs):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gi, gj = r, c
            elif b_idx == 1:
                gi, gj = r, 2 + c
            elif b_idx == 2:
                gi, gj = 2 + r, c
            elif b_idx == 3:
                gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res

    def get_q_element(q_elem_idx, q_block_coeffs):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gj, gk = r, c
            elif b_idx == 1:
                gj, gk = r, 2 + c
            elif b_idx == 2:
                gj, gk = 2 + r, c
            elif b_idx == 3:
                gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res

    for s_idx, (p_bc, q_bc, c_contribs) in enumerate(super_defs):
        for i_idx, (p_elems, q_elems, r_contribs) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []

            for p_elem, coeff in p_elems:
                for gi, gj, c in get_p_element(p_elem, p_bc):
                    u_list.append((gi, gj, coeff * c))

            for q_elem, coeff in q_elems:
                for gj, gk, c in get_q_element(q_elem, q_bc):
                    v_list.append((gj, gk, coeff * c))

            for br, bc, sign in c_contribs:
                for lr, lc, local_coeff in r_contribs:
                    gi = br * 2 + lr
                    gk = bc * 2 + lc
                    w_list.append((gk, gi, sign * local_coeff))

            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1

    # Part 2: 4x4x1 (rank 16) - schoolbook
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1

    # Rank reduction: try to eliminate linearly dependent columns
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Build the tensor as a set of vectors; check for linear dependencies
    # Each column c defines a rank-1 tensor term u_c ⊗ v_c ⊗ w_c
    # If a column is a linear combination of others, we can potentially remove it
    # (but only if the resulting decomposition still satisfies the tensor equation)
    
    # For now, check if any columns are exactly zero or duplicates
    kept = list(range(rank))
    for c in range(rank):
        if time.time() - start_time > DEADLINE - 10:
            break
        # Skip if already removed
        if c not in kept:
            continue
        # Check if column c is a scalar multiple of an earlier kept column
        for d in kept:
            if d >= c:
                break
            # Check if u_c = alpha * u_d, v_c = alpha * v_d, w_c = alpha * w_d
            if np.allclose(U_arr[:, c], 0, atol=1e-12) and np.allclose(V_arr[:, c], 0, atol=1e-12) and np.allclose(W_arr[:, c], 0, atol=1e-12):
                kept.remove(c)
                break
            # Check for exact scalar multiple
            nonzero_u = U_arr[:, c] != 0
            if np.any(nonzero_u):
                idx = np.argmax(nonzero_u)
                if U_arr[idx, d] != 0:
                    alpha = U_arr[idx, c] / U_arr[idx, d]
                    if np.allclose(U_arr[:, c], alpha * U_arr[:, d], atol=1e-12) and \
                       np.allclose(V_arr[:, c], alpha * V_arr[:, d], atol=1e-12) and \
                       np.allclose(W_arr[:, c], alpha * W_arr[:, d], atol=1e-12):
                        kept.remove(c)
                        break
    
    # Rebuild with only kept columns
    if len(kept) < rank:
        new_rank = len(kept)
        U_new = [[0] * new_rank for _ in range(n * m)]
        V_new = [[0] * new_rank for _ in range(m * p)]
        W_new = [[0] * new_rank for _ in range(p * n)]
        for new_c, old_c in enumerate(kept):
            for i in range(n * m):
                U_new[i][new_c] = U[old_c][i] if False else U[i][old_c]
            for j in range(m * p):
                V_new[j][new_c] = V[j][old_c]
            for k in range(p * n):
                W_new[k][new_c] = W[k][old_c]
        U, V, W = U_new, V_new, W_new
        rank = new_rank

    if time.time() - start_time > DEADLINE - 5:
        pass

    return {"U": U, "V": V, "W": W}
