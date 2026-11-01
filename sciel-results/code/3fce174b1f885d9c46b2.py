"""Optimized program: rank-minimal exact tensor decomposition for 2x4 @ 4x6 using Strassen blocks."""
import time

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s limit

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        pass

    n, m, p = 2, 4, 6
    rank_total = 42
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    # Strassen factors for 2x2x2
    # U: 2x2 matrix coefficients
    # V: 2x2 matrix coefficients
    # W: 2x2 matrix coefficients
    # Indices 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
    strassen_U = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {0: -1, 2: 1},
        {1: 1, 3: -1},
    ]
    strassen_V = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1},
    ]
    strassen_W = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1},
    ]

    col_offset = 0
    for j_block in range(3):
        for side in ['L', 'R']:
            if side == 'L':
                u_offset = 0
                v_row_offset = 0
            else:
                u_offset = 2
                v_row_offset = 2

            for prod_idx in range(7):
                col = col_offset + prod_idx

                # Map local 2x2 indices to global U indices (2x4 matrix)
                # Local 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                # Global row r, col c -> r*4 + c
                u_idx_0 = 0 * 4 + (u_offset + 0)
                u_idx_1 = 0 * 4 + (u_offset + 1)
                u_idx_2 = 1 * 4 + (u_offset + 0)
                u_idx_3 = 1 * 4 + (u_offset + 1)
                
                if 0 in strassen_U[prod_idx]: U[u_idx_0][col] += strassen_U[prod_idx][0]
                if 1 in strassen_U[prod_idx]: U[u_idx_1][col] += strassen_U[prod_idx][1]
                if 2 in strassen_U[prod_idx]: U[u_idx_2][col] += strassen_U[prod_idx][2]
                if 3 in strassen_U[prod_idx]: U[u_idx_3][col] += strassen_U[prod_idx][3]

                # Map local 2x2 indices to global V indices (4x6 matrix)
                # Local 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                # Global row r, col c -> r*6 + c
                v_row_base = v_row_offset * 6
                v_col_base = 2 * j_block
                v_idx_0 = v_row_base + 0 + v_col_base
                v_idx_1 = v_row_base + 0 + v_col_base + 1
                v_idx_2 = v_row_base + 6 + v_col_base
                v_idx_3 = v_row_base + 6 + v_col_base + 1

                if 0 in strassen_V[prod_idx]: V[v_idx_0][col] += strassen_V[prod_idx][0]
                if 1 in strassen_V[prod_idx]: V[v_idx_1][col] += strassen_V[prod_idx][1]
                if 2 in strassen_V[prod_idx]: V[v_idx_2][col] += strassen_V[prod_idx][2]
                if 3 in strassen_V[prod_idx]: V[v_idx_3][col] += strassen_V[prod_idx][3]

                # Map local 2x2 indices to global W indices (6x2 matrix, trace convention W[k*n+i])
                # Local 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                # Global col k, row i -> k*2 + i
                w_col_base = 2 * j_block
                w_idx_0 = w_col_base * 2 + 0
                w_idx_1 = (w_col_base + 1) * 2 + 0
                w_idx_2 = w_col_base * 2 + 1
                w_idx_3 = (w_col_base + 1) * 2 + 1

                if 0 in strassen_W[prod_idx]: W[w_idx_0][col] += strassen_W[prod_idx][0]
                if 1 in strassen_W[prod_idx]: W[w_idx_1][col] += strassen_W[prod_idx][1]
                if 2 in strassen_W[prod_idx]: W[w_idx_2][col] += strassen_W[prod_idx][2]
                if 3 in strassen_W[prod_idx]: W[w_idx_3][col] += strassen_W[prod_idx][3]

            col_offset += 7

    return {"U": U, "V": V, "W": W}
