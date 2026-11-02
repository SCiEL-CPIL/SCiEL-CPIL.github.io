"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS          # n=2 rows, m=4 cols, p=8 cols
    rank = 4 * n                  # 4 multiplies per row, 2 rows -> 6

    # 4-point evaluation matrix (maps [a0,a1,a2,a3] -> [f(1),f(-1),f(i),f(-i)])
    M4 = np.array([
        [1.0,  1.0,  1.0,  1.0],
        [1.0, -1.0,  1.0, -1.0],
        [1.0,  1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0,  1.0],
    ])
    # Interpolation matrix (maps [f(1),f(-1),f(i),f(-i)] -> [a0,a1,a2,a3]); M4 * M4.T = 4*I
    M4_inv = (M4.T / 4.0)

    # Build factor matrices
    U = np.zeros((n * m, rank))   # (8, 6)
    V = np.zeros((m * p, rank))   # (32, 6)
    W = np.zeros((p * n, rank))   # (16, 6)

    for i in range(n):
        for t in range(4):
            col = i * 4 + t
            # U: input coefficient (row i, column j) into evaluation t
            for j in range(m):
                U[i * m + j, col] = M4[t, j]
            # V: for each column k, evaluation t of the row-vector [b_{k,0..3}]
            for k in range(p):
                for j in range(m):
                    V[j * p + k, col] = M4[t, j]
            # W: output coefficient (row i, column k) from product-evaluation t
            for k in range(p):
                for j in range(m):
                    W[k * n + i, col] = M4_inv[t, j]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
