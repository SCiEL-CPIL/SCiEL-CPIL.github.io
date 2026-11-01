import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    def _verify(U, V, W, rank):
        """Verify the decomposition using numpy for speed."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

    # Rank 36 decomposition via Strassen on 2x2 blocks
    rank = 36

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    col = 0

    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices

        # M1 = (A00+A11)(B00+B11) -> C00 + C11
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1

        # M2 = (A10+A11)B00 -> C10 - C11
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
        col += 1

        # M3 = A00(B01-B11) -> C01 + C11
        U[u_idx(y00[0], y00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1

        # M4 = A11(B10-B00) -> C00 + C10
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        col += 1

        # M5 = (A00+A01)B11 -> -C00 + C01
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y01[0], y01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        col += 1

        # M6 = (A10-A00)(B00+B01) -> C11
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y00[0], y00[1])][col] = -1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1

        # M7 = (A01-A11)(B10+B11) -> C00
        U[u_idx(y01[0], y01[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = -1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        col += 1

    # Block 1: A_left (cols 0,1) x B_top-left (rows 0,1, cols 0,1) -> C[0:2, 0:2]
    y1 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    z1 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    w1 = [(0, 0), (1, 0), (0, 1), (1, 1)]
    strassen_block(y1, z1, w1)

    # Block 2: A_right (cols 2,3) x B_top-right (rows 2,3, cols 0,1) -> C[0:2, 0:2]
    y2 = [(0, 2), (0, 3), (1, 2), (1, 3)]
    z2 = [(2, 0), (2, 1), (3, 0), (3, 1)]
    w2 = [(0, 0), (1, 0), (0, 1), (1, 1)]
    strassen_block(y2, z2, w2)

    # Block 3: A_left (cols 0,1) x B_bot-left (rows 0,1, cols 2,3) -> C[0:2, 2:4]
    y3 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    z3 = [(0, 2), (0, 3), (1, 2), (1, 3)]
    w3 = [(2, 0), (3, 0), (2, 1), (3, 1)]
    strassen_block(y3, z3, w3)

    # Block 4: A_right (cols 2,3) x B_bot-right (rows 2,3, cols 2,3) -> C[0:2, 2:4]
    y4 = [(0, 2), (0, 3), (1, 2), (1, 3)]
    z4 = [(2, 2), (2, 3), (3, 2), (3, 3)]
    w4 = [(2, 0), (3, 0), (2, 1), (3, 1)]
    strassen_block(y4, z4, w4)

    # Block 5: A_left (cols 0,1) x B_col4 (rows 0,1, col 4) -> C[0:2, 4]
    for ii in range(2):
        for jj in range(2):
            U[u_idx(ii, jj)][col] = 1
            V[v_idx(jj, 4)][col] = 1
            W[w_idx(4, ii)][col] += 1
            col += 1

    # Block 6: A_right (cols 2,3) x B_col4 (rows 2,3, col 4) -> C[0:2, 4]
    for ii in range(2):
        for jj in range(2):
            U[u_idx(ii, jj + 2)][col] = 1
            V[v_idx(jj + 2, 4)][col] = 1
            W[w_idx(4, ii)][col] += 1
            col += 1

    # Ensure rank matches actual columns used
    assert col == rank, f"Column count {col} != rank {rank}"

    # Verify the decomposition
    if time.time() < deadline:
        if not _verify(U, V, W, rank):
            # Fallback: schoolbook rank 40
            rank = 40
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[u_idx(i, j)][col] = 1
                        V[v_idx(j, k)][col] = 1
                        W[w_idx(k, i)][col] = 1
                        col += 1

    return {"U": U, "V": V, "W": W}
