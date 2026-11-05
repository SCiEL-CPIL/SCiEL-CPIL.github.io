import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    N_U = n * m  # 6
    N_V = m * p  # 21
    N_W = p * n  # 14

    start_time = time.time()
    deadline = start_time + 110.0

    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def build_rank38():
        """Rank-38 decomposition using Strassen for 2x2 sub-blocks.
        
        Partition the 2x3x7 problem:
        - Split A into A_left (2x2, cols 0,1) and A_right (2x1, col 2)
        - Split B into B_top (2x7, rows 0,1) and B_bot (1x7, row 2)
        - C = A_left @ B_top + A_right @ B_bot
        
        For A_left @ B_top (2x2x7):
        - Split B_top into B_left (2x3, cols 0-2) and B_right (2x4, cols 3-6)
        - B_left has 3 cols: use Strassen on cols (0,1) -> rank 7, col 2 schoolbook -> rank 4
        - B_right has 4 cols: use Strassen on cols (3,4) -> rank 7, cols (5,6) -> rank 7
        - Total for A_left @ B_top: 7 + 4 + 7 + 7 = 25
        
        For A_right @ B_bot (2x1x7):
        - Schoolbook: 2 * 1 * 7 = 14
        
        Total: 25 + 14 = 39... let me reconsider.
        
        Actually, let me think differently. The 2x2x2 Strassen uses rank 7.
        
        Alternative: partition into 2x2x3 and 2x2x4 and 2x1x7.
        - 2x2x3: Strassen on first 2 k's (rank 7) + schoolbook on 3rd k (rank 4) = 11
        - 2x2x4: Strassen on k-pair (3,4) (rank 7) + Strassen on k-pair (5,6) (rank 7) = 14
        - 2x1x7: schoolbook = 14
        Total: 11 + 14 + 14 = 39
        
        To get rank 38, I need to save 1 more. Let me think about combining.
        
        Actually, the j=2 block contributes 14 terms (2 i's * 1 j * 7 k's).
        The Strassen blocks for j=0,1 contribute 25 terms (7*3 + 4 = 25 for 3 Strassen blocks of 2x2x2 plus one schoolbook 2x2x1).
        
        Wait, let me recount. For j=0,1:
        - 3 Strassen blocks of 2x2x2 (k-pairs (0,1), (2,3), (4,5)): 3*7 = 21
        - 1 schoolbook 2x2x1 (k=6): 4
        Total for j=0,1: 25
        
        For j=2: 14
        Grand total: 39
        
        To get 38, I need to find a way to save 1 term. This might not be straightforward.
        
        Let me try a different approach: use a 2x2x3 Strassen-like decomposition.
        The 2x2x3 matrix multiplication tensor has known rank. Let me check if it can be done in rank 10 instead of 11.
        
        Actually, for now let me just implement the rank-39 solution cleanly and verify it works.
        """
        rank = 39
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]

        col = 0

        def add_term(u_idx_coeff, v_idx_coeff, w_idx_coeff):
            nonlocal col
            for idx, c in u_idx_coeff:
                U[idx][col] += c
            for idx, c in v_idx_coeff:
                V[idx][col] += c
            for idx, c in w_idx_coeff:
                W[idx][col] += c
            col += 1

        def strassen_2x2x2(k0, k1):
            """Add 7 Strassen terms for 2x2x2 block with k indices k0, k1."""
            a00, a01, a10, a11 = 0, 1, 3, 4
            b00, b01, b10, b11 = k0, k0 + 1, 7 + k0, 7 + k0 + 1
            c00, c01, c10, c11 = 2 * k0, 2 * k0 + 1, 2 * k1, 2 * k1 + 1

            add_term([(a00, 1), (a11, 1)], [(b00, 1), (b11, 1)], [(c00, 1), (c11, 1)])
            add_term([(a10, 1), (a11, 1)], [(b00, 1)], [(c10, 1), (c11, -1)])
            add_term([(a00, 1)], [(b01, 1), (b11, -1)], [(c01, 1), (c11, 1)])
            add_term([(a11, 1)], [(b10, 1), (b00, -1)], [(c00, 1), (c10, 1)])
            add_term([(a00, 1), (a01, 1)], [(b11, 1)], [(c00, -1), (c01, 1)])
            add_term([(a10, 1), (a00, -1)], [(b00, 1), (b01, 1)], [(c11, 1)])
            add_term([(a01, 1), (a11, -1)], [(b10, 1), (b11, 1)], [(c00, 1)])

        # 3 Strassen blocks for k-pairs (0,1), (2,3), (4,5)
        strassen_2x2x2(0, 1)
        strassen_2x2x2(2, 3)
        strassen_2x2x2(4, 5)

        # Schoolbook for k=6, j=0,1: 4 terms
        b06, b16 = 6, 7 + 6
        c06, c16 = 2 * 6, 2 * 6 + 1
        add_term([(0, 1)], [(b06, 1)], [(c06, 1)])
        add_term([(1, 1)], [(b16, 1)], [(c06, 1)])
        add_term([(3, 1)], [(b06, 1)], [(c16, 1)])
        add_term([(4, 1)], [(b16, 1)], [(c16, 1)])

        # Schoolbook for j=2: 14 terms
        for k in range(7):
            b2k = 14 + k
            c0k, c1k = 2 * k, 2 * k + 1
            add_term([(2, 1)], [(b2k, 1)], [(c0k, 1)])
            add_term([(5, 1)], [(b2k, 1)], [(c1k, 1)])

        return {"U": U, "V": V, "W": W}

    def verify_decomposition(result):
        U = np.array(result["U"], dtype=np.int64)
        V = np.array(result["V"], dtype=np.int64)
        W = np.array(result["W"], dtype=np.int64)
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        T_target = np.zeros((N_U, N_V, N_W), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i * m + j, j * p + k, k * n + i] = 1
        return np.array_equal(T_calc, T_target)

    if time.time() > deadline - 10:
        return get_schoolbook()

    result = build_rank38()

    if verify_decomposition(result):
        return result

    return get_schoolbook()
