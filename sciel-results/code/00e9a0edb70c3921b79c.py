"""
6x6 matrix multiplication tensor decomposition using 3x3 grid of 2x2 Strassen blocks.
Rank = 27 * 7 = 189.
"""
import time

DIMENSIONS = [6, 6, 6]

_DEADLINE = 140.0

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS

    block_size = 2
    blocks_per_dim = 3

    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen patterns for 2x2 matrices.
    # Each pattern: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict (local row, local col) -> value  (for A block)
    # v_coeffs: dict (local row, local col) -> value  (for B block)
    # w_coeffs: dict (local row, local col) -> value  (for C block)
    strassen_patterns = [
        # M0 = (A00+A11)(B00+B11)
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1},
         {(0, 0): 1, (1, 1): 1}),
        # M1 = (A10+A11)B00
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1},
         {(1, 0): 1, (1, 1): -1}),
        # M2 = A00(B01-B11)
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1},
         {(0, 1): 1, (1, 1): 1}),
        # M3 = A11(B10-B00)
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1},
         {(0, 0): 1, (1, 0): 1}),
        # M4 = (A00+A01)B11
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1},
         {(0, 0): -1, (0, 1): 1}),
        # M5 = (A10-A00)(B00+B01)
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1},
         {(1, 1): 1}),
        # M6 = (A01-A11)(B10+B11)
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1},
         {(0, 0): 1}),
    ]

    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    # Update U: A is n x m, index r*m + c
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val

                    # Update V: B is m x p, index r*p + c
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val

                    # Update W: C is n x p, index k*n + i
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val

                    col_idx += 1

                    if time.time() - t0 > _DEADLINE:
                        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
