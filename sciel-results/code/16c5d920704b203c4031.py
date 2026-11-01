"""Matrix multiplication 2x2 by 2x14: rank-minimal exact tensor decomposition."""
import time
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _DEADLINE = 120.0
    n, m, p = DIMENSIONS  # 2, 2, 14
    num_blocks = p // 2  # 7 blocks of 2x2

    rank = num_blocks * 7

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen_terms = [
        (1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1),
        (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, -1),
        (1, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0, 1),
        (0, 0, 0, 1, -1, 0, 1, 0, 1, 0, 1, 0),
        (1, 1, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0),
        (-1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1),
        (0, 1, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0),
    ]

    for b in range(num_blocks):
        base_col = b * 7
        base_k = 2 * b
        for t, term in enumerate(strassen_terms):
            col = base_col + t
            a_c, b_c, c_c, d_c, e_c, f_c, g_c, h_c, c11_c, c12_c, c21_c, c22_c = term

            U[0][col] = a_c
            U[1][col] = b_c
            U[2][col] = c_c
            U[3][col] = d_c

            V[base_k][col] = e_c
            V[base_k + 1][col] = f_c
            V[14 + base_k][col] = g_c
            V[14 + base_k + 1][col] = h_c

            W[base_k * 2][col] = c11_c
            W[base_k * 2 + 2][col] = c12_c
            W[base_k * 2 + 1][col] = c21_c
            W[base_k * 2 + 3][col] = c22_c

    if time.time() - _t0 > _DEADLINE - 5:
        pass
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
