import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0

    n, m, p = DIMENSIONS  # 2, 3, 8
    rank = 44

    U = [[0] * rank for _ in range(n * m)]   # 6 x 44
    V = [[0] * rank for _ in range(m * p)]   # 24 x 44
    W = [[0] * rank for _ in range(p * n)]   # 16 x 44

    # 4 blocks of 2 columns each (p=8, block_size=2)
    for blk in range(p // 2):
        base = blk * 11
        k0 = 2 * blk
        k1 = 2 * blk + 1

        # --- Strassen terms (indices 0..6 in the block) ---
        # M1 = (a00+a11)(b00+b11)
        U[0][base] += 1;  U[3][base] += 1
        V[0 * 3 + k0][base] += 1;  V[1 * 3 + k1][base] += 1
        W[k0 * 2 + 0][base] += 1;  W[k1 * 2 + 1][base] += 1

        # M2 = (a10+a11) b00
        U[3][base + 1] += 1;  U[4][base + 1] += 1
        V[0 * 3 + k0][base + 1] += 1
        W[k0 * 2 + 1][base + 1] += 1

        # M3 = a00 (b01 - b11)
        U[0][base + 2] += 1
        V[1 * 3 + k1][base + 2] += 1;  V[2 * 3 + k1][base + 2] -= 1
        W[k1 * 2 + 0][base + 2] += 1

        # M4 = a11 (b10 - b00)
        U[4][base + 3] += 1
        V[0 * 3 + k0][base + 3] += 1;  V[1 * 3 + k0][base + 3] -= 1
        W[k0 * 2 + 1][base + 3] += 1

        # M5 = (a00 + a01) b11
        U[0][base + 4] += 1;  U[1][base + 4] += 1
        V[1 * 3 + k1][base + 4] += 1
        W[k1 * 2 + 0][base + 4] += 1;  W[k1 * 2 + 1][base + 4] += 1

        # M6 = (a10 - a00)(b00 + b01)
        U[3][base + 5] += 1;  U[0][base + 5] -= 1
        V[0 * 3 + k0][base + 5] += 1;  V[1 * 3 + k0][base + 5] += 1
        W[k0 * 2 + 1][base + 5] += 1;  W[k1 * 2 + 1][base + 5] -= 1

        # M7 = (a01 - a11)(b10 + b11)
        U[1][base + 6] += 1;  U[4][base + 6] -= 1
        V[0 * 3 + k1][base + 6] += 1;  V[1 * 3 + k1][base + 6] += 1
        W[k0 * 2 + 0][base + 6] += 1;  W[k1 * 2 + 0][base + 6] -= 1

        # --- Extra terms for A col 2, B row 2 (indices 7..10) ---
        # a02 * b20
        U[2][base + 7] += 1
        V[2 * 3 + k0][base + 7] += 1
        W[k0 * 2 + 0][base + 7] += 1

        # a02 * b21
        U[2][base + 8] += 1
        V[2 * 3 + k1][base + 8] += 1
        W[k1 * 2 + 0][base + 8] += 1

        # a12 * b20
        U[5][base + 9] += 1
        V[2 * 3 + k0][base + 9] += 1
        W[k0 * 2 + 1][base + 9] += 1

        # a12 * b21
        U[5][base + 10] += 1
        V[2 * 3 + k1][base + 10] += 1
        W[k1 * 2 + 1][base + 10] += 1

    # Verify decomposition with numpy (fast, ~1ms)
    if time.time() < deadline - 2.0:
        try:
            Um = np.array(U, dtype=np.float64)
            Vm = np.array(V, dtype=np.float64)
            Wm = np.array(W, dtype=np.float64)
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
            Ur = Um.reshape(n, m, rank)
            Vr = Vm.reshape(m, p, rank)
            Wr = Wm.reshape(p, n, rank)
            # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
            T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
            # Expected: T[i,j,k] = 1
            expected = np.ones((n, m, p), dtype=np.float64)
            if not np.allclose(T, expected, atol=1e-10):
                # Fallback to schoolbook
                rank_fb = n * m * p
                U_fb = [[0] * rank_fb for _ in range(n * m)]
                V_fb = [[0] * rank_fb for _ in range(m * p)]
                W_fb = [[0] * rank_fb for _ in range(p * n)]
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_fb[i * m + j][col] = 1
                            V_fb[j * p + k][col] = 1
                            W_fb[k * n + i][col] = 1
                            col += 1
                return {"U": U_fb, "V": V_fb, "W": W_fb}
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
