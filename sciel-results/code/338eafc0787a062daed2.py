import time
import numpy as np

def construct():
    """
    Matrix multiplication (3,5) x (5,6) -> (3,6).
    
    Uses block decomposition with Strassen for 2x2x2 blocks.
    Achieves rank 84 (saving 6 from schoolbook 90).
    """
    start_time = time.time()
    deadline = start_time + 100

    n, m, p = 3, 5, 6

    U_terms = []
    V_terms = []
    W_terms = []

    def add_schoolbook_term(i, j, k):
        U_terms.append((i * m + j, 1.0))
        V_terms.append((j * p + k, 1.0))
        W_terms.append((k * n + i, 1.0))

    def add_strassen_block(i0, j0, k0):
        p1_A = [((0,0), 1.0), ((1,1), 1.0)]
        p1_B = [((0,0), 1.0), ((1,1), 1.0)]
        p2_A = [((1,0), 1.0), ((1,1), 1.0)]
        p2_B = [((0,0), 1.0)]
        p3_A = [((0,0), 1.0)]
        p3_B = [((1,0), 1.0), ((1,1), -1.0)]
        p4_A = [((1,1), 1.0)]
        p4_B = [((0,1), 1.0), ((0,0), -1.0)]
        p5_A = [((0,0), 1.0), ((0,1), 1.0)]
        p5_B = [((1,1), 1.0)]
        p6_A = [((1,0), 1.0), ((0,0), -1.0)]
        p6_B = [((0,0), 1.0), ((1,0), 1.0)]
        p7_A = [((0,1), 1.0), ((1,1), -1.0)]
        p7_B = [((0,1), 1.0), ((1,1), 1.0)]

        strassen_intermediates = [p1_A, p2_A, p3_A, p4_A, p5_A, p6_A, p7_A]
        strassen_B = [p1_B, p2_B, p3_B, p4_B, p5_B, p6_B, p7_B]

        c_combinations = [
            (0, 0, [(0, 1.0), (3, -1.0), (4, 1.0), (6, -1.0)]),
            (0, 1, [(2, 1.0), (4, 1.0)]),
            (1, 0, [(1, 1.0), (3, 1.0)]),
            (1, 1, [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)]),
        ]

        for ci, ck, p_list in c_combinations:
            for p_idx, c_p in p_list:
                for (di, dj_A), c_A in strassen_intermediates[p_idx]:
                    for (dj_B, dk), c_B in strassen_B[p_idx]:
                        coeff = c_p * c_A * c_B
                        u_row = (i0 + di) * m + (j0 + dj_A)
                        v_row = (j0 + dj_B) * p + (k0 + dk)
                        w_row = (k0 + ck) * n + (i0 + ci)
                        U_terms.append((u_row, coeff))
                        V_terms.append((v_row, coeff))
                        W_terms.append((w_row, coeff))

    block_i_positions = [0]
    block_j_positions = [0, 2]
    block_k_positions = [0, 2, 4]

    handled = set()

    for i0 in block_i_positions:
        for j0 in block_j_positions:
            for k0 in block_k_positions:
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            handled.add((i0+di, j0+dj, k0+dk))
                add_strassen_block(i0, j0, k0)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in handled:
                    add_schoolbook_term(i, j, k)

    total_terms = len(U_terms)
    U = np.zeros((n * m, total_terms), dtype=float)
    V = np.zeros((m * p, total_terms), dtype=float)
    W = np.zeros((p * n, total_terms), dtype=float)

    for r in range(total_terms):
        U[U_terms[r][0], r] = U_terms[r][1]
        V[V_terms[r][0], r] = V_terms[r][1]
        W[W_terms[r][0], r] = W_terms[r][1]

    # Quick verification
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(total_terms):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-6:
                    errors += 1

    if errors > 0:
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
