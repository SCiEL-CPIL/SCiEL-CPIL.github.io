"""
Matrix multiplication tensor decomposition for 2x5 * 5x6.

Strategy:
1. Use Strassen's algorithm on 2x2 blocks of the middle dimension (j) and
   output dimension (k).
2. Partition j into pairs (0,1), (2,3) and handle j=4 separately.
3. Partition k into pairs (0,1), (2,3), (4,5).
4. For each (j-pair, k-pair), use 7 Strassen products instead of 4 schoolbook.
5. For j=4, use 6 schoolbook products (2 rows x 6 cols).
6. Total rank: 2 * 3 * 7 + 2 * 6 = 42 + 12 = 54.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin before 120s limit

    # Rank 54 decomposition using Strassen's algorithm on 2x2x2 blocks
    # Blocks: j in {0,1}, {2,3} and k in {0,1}, {2,3}, {4,5}
    # Remaining: j=4, k in 0..5

    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]

    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # 7 products for Strassen
            # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
            U[0 * m + j0][col] += 1
            U[1 * m + j1][col] += 1
            V[j0 * p + k0][col] += 1
            V[j1 * p + k1][col] += 1
            W[k0 * n + 0][col] += 1
            W[k1 * n + 1][col] += 1
            col += 1

            # P2 = (A_1,j0 + A_1,j1) B_j0,k0
            U[1 * m + j0][col] += 1
            U[1 * m + j1][col] += 1
            V[j0 * p + k0][col] += 1
            W[k0 * n + 1][col] += 1
            W[k1 * n + 1][col] -= 1
            col += 1

            # P3 = A_0,j0 (B_j0,k1 - B_j1,k1)
            U[0 * m + j0][col] += 1
            V[j0 * p + k1][col] += 1
            V[j1 * p + k1][col] -= 1
            W[k1 * n + 0][col] += 1
            W[k1 * n + 1][col] += 1
            col += 1

            # P4 = A_1,j1 (B_j1,k0 - B_j0,k0)
            U[1 * m + j1][col] += 1
            V[j1 * p + k0][col] += 1
            V[j0 * p + k0][col] -= 1
            W[k0 * n + 0][col] += 1
            W[k0 * n + 1][col] += 1
            col += 1

            # P5 = (A_0,j0 + A_0,j1) B_j1,k1
            U[0 * m + j0][col] += 1
            U[0 * m + j1][col] += 1
            V[j1 * p + k1][col] += 1
            W[k0 * n + 0][col] -= 1
            W[k1 * n + 0][col] += 1
            col += 1

            # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
            U[1 * m + j0][col] += 1
            U[0 * m + j0][col] -= 1
            V[j0 * p + k0][col] += 1
            V[j0 * p + k1][col] += 1
            W[k1 * n + 1][col] += 1
            col += 1

            # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
            U[0 * m + j1][col] += 1
            U[1 * m + j1][col] -= 1
            V[j1 * p + k0][col] += 1
            V[j1 * p + k1][col] += 1
            W[k0 * n + 0][col] += 1
            col += 1

    # Remaining j=4
    for k in range(p):
        # C_0,k = A_0,4 B_4,k
        U[0 * m + 4][col] += 1
        V[4 * p + k][col] += 1
        W[k * n + 0][col] += 1
        col += 1

        # C_1,k = A_1,4 B_4,k
        U[1 * m + 4][col] += 1
        V[4 * p + k][col] += 1
        W[k * n + 1][col] += 1
        col += 1

    # Verify the decomposition
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(
                    U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
                    for c in range(rank)
                )
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    # If verification fails (shouldn't happen), fall back to schoolbook
    if not valid:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
