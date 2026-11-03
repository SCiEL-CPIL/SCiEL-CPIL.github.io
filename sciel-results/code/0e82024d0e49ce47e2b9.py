import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    """
    Constructs a rank-28 decomposition for 2x4 * 4x4 matrix multiplication
    using 4 independent Strassen (rank-7) blocks.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col
        u_00 = a_rows[0] * m + a_cols[0]
        u_01 = a_rows[0] * m + a_cols[1]
        u_10 = a_rows[1] * m + a_cols[0]
        u_11 = a_rows[1] * m + a_cols[1]
        v_00 = b_rows[0] * p + b_cols[0]
        v_01 = b_rows[0] * p + b_cols[1]
        v_10 = b_rows[1] * p + b_cols[0]
        v_11 = b_rows[1] * p + b_cols[1]
        w_00 = c_cols[0] * n + c_rows[0]
        w_01 = c_cols[1] * n + c_rows[0]
        w_10 = c_cols[0] * n + c_rows[1]
        w_11 = c_cols[1] * n + c_rows[1]

        # M1 = (P00+P11)*(Q00+Q11)
        U[u_00][col] += 1; U[u_11][col] += 1
        V[v_00][col] += 1; V[v_11][col] += 1
        W[w_00][col] += 1; W[w_11][col] += 1
        col += 1
        # M2 = (P10+P11)*Q00
        U[u_10][col] += 1; U[u_11][col] += 1
        V[v_00][col] += 1
        W[w_10][col] += 1; W[w_11][col] += -1
        col += 1
        # M3 = P00*(Q01-Q11)
        U[u_00][col] += 1
        V[v_01][col] += 1; V[v_11][col] += -1
        W[w_01][col] += 1; W[w_11][col] += 1
        col += 1
        # M4 = P11*(Q10-Q00)
        U[u_11][col] += 1
        V[v_10][col] += 1; V[v_00][col] += -1
        W[w_00][col] += 1; W[w_10][col] += 1
        col += 1
        # M5 = (P00+P01)*Q11
        U[u_00][col] += 1; U[u_01][col] += 1
        V[v_11][col] += 1
        W[w_01][col] += 1; W[w_00][col] += -1
        col += 1
        # M6 = (P10-P00)*(Q00+Q01)
        U[u_10][col] += 1; U[u_00][col] += -1
        V[v_00][col] += 1; V[v_01][col] += 1
        W[w_11][col] += 1
        col += 1
        # M7 = (P01-P11)*(Q10+Q11)
        U[u_01][col] += 1; U[u_11][col] += -1
        V[v_10][col] += 1; V[v_11][col] += 1
        W[w_00][col] += 1
        col += 1

    add_strassen_block([0,1],[0,1],[0,1],[0,1],[0,1],[0,1])
    add_strassen_block([0,1],[2,3],[2,3],[0,1],[0,1],[0,1])
    add_strassen_block([0,1],[0,1],[0,1],[2,3],[0,1],[2,3])
    add_strassen_block([0,1],[2,3],[2,3],[2,3],[0,1],[2,3])

    assert col == rank
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
