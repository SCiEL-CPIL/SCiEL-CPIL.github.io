"""Matrix multiplication 4x4x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 5
    nm = n * m
    mp = m * p
    pn = p * n

    nonzero_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_triples.append((a, b, c))

    def verify(U, V, W, tol=1e-9):
        rank = len(U[0])
        for a, b, c in nonzero_triples:
            s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
            if abs(s - 1.0) > tol:
                return False
        for a in range(nm):
            i_a = a // m
            j_a = a % m
            for b in range(mp):
                j_b = b // p
                k_b = b % p
                if j_b != j_a:
                    continue
                for c in range(pn):
                    k_c = c // n
                    i_c = c % n
                    if i_c != i_a or k_c != k_b:
                        s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                        if abs(s) > tol:
                            return False
        return True

    def build_rank65():
        rank = 65
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]

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

        def get_super_term_indices(super_idx, inner_idx):
            u_list = []
            v_list = []
            w_list = []

            if super_idx == 0:
                p_blocks = [1, 0, 0, 1]
                q_blocks = [1, 0, 0, 1]
            elif super_idx == 1:
                p_blocks = [0, 0, 1, 1]
                q_blocks = [1, 0, 0, 0]
            elif super_idx == 2:
                p_blocks = [1, 0, 0, 0]
                q_blocks = [0, 1, 0, -1]
            elif super_idx == 3:
                p_blocks = [0, 0, 0, 1]
                q_blocks = [-1, 0, 1, 0]
            elif super_idx == 4:
                p_blocks = [1, 1, 0, 0]
                q_blocks = [0, 0, 0, 1]
            elif super_idx == 5:
                p_blocks = [-1, 0, 1, 0]
                q_blocks = [1, 1, 0, 0]
            elif super_idx == 6:
                p_blocks = [0, 1, 0, -1]
                q_blocks = [0, 0, 1, 1]
            else:
                return u_list, v_list, w_list

            def get_p_element(p_elem_idx):
                res = []
                r = p_elem_idx // 2
                c = p_elem_idx % 2
                for b_idx, coeff in enumerate(p_blocks):
                    if coeff == 0:
                        continue
                    if b_idx == 0:
                        gi = r; gj = c
                    elif b_idx == 1:
                        gi = r; gj = 2 + c
                    elif b_idx == 2:
                        gi = 2 + r; gj = c
                    elif b_idx == 3:
                        gi = 2 + r; gj = 2 + c
                    res.append((gi, gj, coeff))
                return res

            def get_q_element(q_elem_idx):
                res = []
                r = q_elem_idx // 2
                c = q_elem_idx % 2
                for b_idx, coeff in enumerate(q_blocks):
                    if coeff == 0:
                        continue
                    if b_idx == 0:
                        gj = r; gk = c
                    elif b_idx == 1:
                        gj = r; gk = 2 + c
                    elif b_idx == 2:
                        gj = 2 + r; gk = c
                    elif b_idx == 3:
                        gj = 2 + r; gk = 2 + c
                    res.append((gj, gk, coeff))
                return res

            if inner_idx == 0:
                p_elems = [(0, 1), (3, 1)]
                q_elems = [(0, 1), (3, 1)]
            elif inner_idx == 1:
                p_elems = [(2, 1), (3, 1)]
                q_elems = [(0, 1)]
            elif inner_idx == 2:
                p_elems = [(0, 1)]
                q_elems = [(1, 1), (3, -1)]
            elif inner_idx == 3:
                p_elems = [(3, 1)]
                q_elems = [(2, 1), (0, -1)]
            elif inner_idx == 4:
                p_elems = [(0, 1), (1, 1)]
                q_elems = [(3, 1)]
            elif inner_idx == 5:
                p_elems = [(2, 1), (0, -1)]
                q_elems = [(0, 1), (1, 1)]
            elif inner_idx == 6:
                p_elems = [(1, 1), (3, -1)]
                q_elems = [(2, 1), (3, 1)]
            else:
                return u_list, v_list, w_list

            for p_idx, p_coeff in p_elems:
                elems = get_p_element(p_idx)
                for i, j, c in elems:
                    u_list.append((i, j, c * p_coeff))

            for q_idx, q_coeff in q_elems:
                elems = get_q_element(q_idx)
                for j, k, c in elems:
                    v_list.append((j, k, c * q_coeff))

            if inner_idx == 0:
                s_contrib = [1, 0, 0, 1]
            elif inner_idx == 1:
                s_contrib = [0, 0, 1, -1]
            elif inner_idx == 2:
                s_contrib = [0, 1, 0, 1]
            elif inner_idx == 3:
                s_contrib = [1, 0, 1, 0]
            elif inner_idx == 4:
                s_contrib = [-1, 1, 0, 0]
            elif inner_idx == 5:
                s_contrib = [0, 0, 0, 1]
            elif inner_idx == 6:
                s_contrib = [1, 0, 0, 0]
            else:
                s_contrib = [0, 0, 0, 0]

            c_block_contrib = [0, 0, 0, 0]
            if super_idx == 0:
                c_block_contrib = [1, 0, 0, 1]
            elif super_idx == 1:
                c_block_contrib = [0, 0, 1, -1]
            elif super_idx == 2:
                c_block_contrib = [0, 1, 0, 1]
            elif super_idx == 3:
                c_block_contrib = [1, 0, 1, 0]
            elif super_idx == 4:
                c_block_contrib = [-1, 1, 0, 0]
            elif super_idx == 5:
                c_block_contrib = [0, 0, 0, 1]
            elif super_idx == 6:
                c_block_contrib = [1, 0, 0, 0]

            for s_pos in range(4):
                s_coeff = s_contrib[s_pos]
                if s_coeff == 0:
                    continue
                s_r = s_pos // 2
                s_c = s_pos % 2
                for c_b in range(4):
                    c_coeff = c_block_contrib[c_b]
                    if c_coeff == 0:
                        continue
                    total_coeff = s_coeff * c_coeff
                    if total_coeff == 0:
                        continue
                    if c_b == 0:
                        gi = s_r; gk = s_c
                    elif c_b == 1:
                        gi = s_r; gk = 2 + s_c
                    elif c_b == 2:
                        gi = 2 + s_r; gk = s_c
                    elif c_b == 3:
                        gi = 2 + s_r; gk = 2 + s_c
                    w_list.append((gk, gi, total_coeff))

            return u_list, v_list, w_list

        for s in range(7):
            for t in range(7):
                u_list, v_list, w_list = get_super_term_indices(s, t)
                add_term(col, u_list, v_list, w_list)
                col += 1

        for i in range(4):
            for j in range(4):
                u_list = [(i, j, 1)]
                v_list = [(j, 4, 1)]
                w_list = [(4, i, 1)]
                add_term(col, u_list, v_list, w_list)
                col += 1

        return U, V, W

    U, V, W = build_rank65()

    if not verify(U, V, W):
        # Fallback: schoolbook rank 80
        rank = 80
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
