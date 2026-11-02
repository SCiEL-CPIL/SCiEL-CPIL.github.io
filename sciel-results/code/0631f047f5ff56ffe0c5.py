import numpy as np

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_total = 25
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Local Strassen matrices
    # U_local: 4x7, V_local: 4x7, W_local: 4x7
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, -1],
        [1, 1, 0, 0, 0, 0, 1]
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_strassen = [
        [1, 0, 0, -1, 1, 0, -1],
        [0, 1, 0, 1, 0, -1, -1],
        [0, 0, 1, 1, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Local Schoolbook matrices for 2x2x1
    # U_local: 4x4, V_local: 2x4, W_local: 2x4
    U_school = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_school = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_school = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    def add_strassen_block(k_offset, col_offset):
        # k_offset: 0, 2, 4
        # col_offset: 0, 7, 14
        for i in range(n):
            for j in range(m):
                u_row = i * m + j
                for c in range(7):
                    U[u_row][col_offset + c] += U_strassen[u_row][c]
                    
                # V rows: j * p + k
                # k = k_offset + 0, k_offset + 1
                for c in range(7):
                    v_row_0 = j * p + (k_offset)
                    v_row_1 = j * p + (k_offset + 1)
                    V[v_row_0][col_offset + c] += V_strassen[j * 2 + 0][c]
                    V[v_row_1][col_offset + c] += V_strassen[j * 2 + 1][c]
                    
                # W rows: k * n + i
                # k = k_offset + 0, k_offset + 1
                for c in range(7):
                    w_row_0 = (k_offset) * n + i
                    w_row_1 = (k_offset + 1) * n + i
                    W[w_row_0][col_offset + c] += W_strassen[0 * 2 + i][c]
                    W[w_row_1][col_offset + c] += W_strassen[1 * 2 + i][c]

    def add_school_block(k_offset, col_offset):
        # k_offset: 6
        # col_offset: 21
        for i in range(n):
            for j in range(m):
                u_row = i * m + j
                for c in range(4):
                    U[u_row][col_offset + c] += U_school[u_row][c]
                    
                # V rows: j * p + k
                # k = k_offset
                for c in range(4):
                    v_row = j * p + k_offset
                    V[v_row][col_offset + c] += V_school[j][c]
                    
                # W rows: k * n + i
                # k = k_offset
                for c in range(4):
                    w_row = k_offset * n + i
                    W[w_row][col_offset + c] += W_school[i][c]

    add_strassen_block(0, 0)
    add_strassen_block(2, 7)
    add_strassen_block(4, 14)
    add_school_block(6, 21)
    
    return {"U": U, "V": V, "W": W}
