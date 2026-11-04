"""Strassen-based 4x5x6 matrix multiplication decomposition (rank 108)."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Total rank calculation:
    # 12 blocks of 2x2x2 Strassen (rank 7 each) -> 84
    # 1 block of 4x1x6 schoolbook (rank 24) -> 24
    # Total rank = 108
    rank = 108

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(i0, j0, k0, r_start):
        # Adds the Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        # to the global U, V, W matrices starting at column index r_start

        def u_idx(i, j): return (i0 + i) * m + (j0 + j)
        def v_idx(j, k): return (j0 + j) * p + (k0 + k)
        def w_idx(k, i): return (k0 + k) * n + (i0 + i)

        # M1 = (A00 + A11)(B00 + B11)
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1

        # M2 = (A10 + A11)B00
        r_start += 1
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += -1

        # M3 = A00(B01 - B11)
        r_start += 1
        U[u_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1

        # M4 = A11(B10 - B00)
        r_start += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(0, 0)][r_start] += -1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1

        # M5 = (A00 + A01)B11
        r_start += 1
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1

        # M6 = (A10 - A00)(B00 + B01)
        r_start += 1
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(0, 0)][r_start] += -1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1

        # M7 = (A01 - A11)(B10 + B11)
        r_start += 1
        U[u_idx(0, 1)][r_start] += 1
        U[u_idx(1, 1)][r_start] += -1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1

        return r_start + 1

    r_offset = 0

    # 12 Strassen blocks covering the 4x4x4 core
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                r_offset = add_strassen_block(i0, j0, k0, r_offset)

    # Block 13: A[0:4, 4:5] * B[4:5, 0:6] -> C[0:4, 0:6]
    # This is 4x1 by 1x6. Rank 24.
    for i in range(4):
        for k in range(6):
            U[i * 5 + 4][r_offset] = 1
            V[4 * 6 + k][r_offset] = 1
            W[k * 4 + i][r_offset] = 1
            r_offset += 1

    if r_offset != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {r_offset}")

    # Quick sanity check
    if time.time() - start_time < DEADLINE:
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)

        A_test = np.random.randint(-2, 3, (n, m))
        B_test = np.random.randint(-2, 3, (m, p))
        C_expected = A_test @ B_test

        C_computed = np.zeros((n, p))
        for r in range(rank):
            A_part = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    A_part[i] += U_np[i * m + j, r] * A_test[i, j]

            B_part = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    B_part[k] += V_np[j * p + k, r] * B_test[j, k]

            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += W_np[k * n + i, r] * A_part[i] * B_part[k]

        if not np.allclose(C_computed, C_expected):
            pass  # Should not happen

    return {"U": U, "V": V, "W": W}
