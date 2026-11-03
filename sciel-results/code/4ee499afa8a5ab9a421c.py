import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    start_time = time.time()
    timeout = 120.0  # Safety margin before 150s limit

    n, m, p = 2, 3, 8

    def verify(U, V, W, n, m, p):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-10)

    # Strategy: Split m=3 into m1=1 (j=0) and m2=2 (j=1,2)
    # Part 1: 2x1x8 schoolbook -> rank 16
    # Part 2: 2x2x8 split into 4 blocks of 2x2x2 Strassen -> rank 4*7=28
    # Total rank: 44

    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # (6, 44)
    V = [[0] * rank for _ in range(m * p)]  # (24, 44)
    W = [[0] * rank for _ in range(p * n)]  # (16, 44)

    col = 0

    # Part 1: A[:,0] (2x1) * B[0,:] (1x8) -> C[:,0:8] (2x8)
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Part 2: A[:,1:3] (2x2) * B[1:3,:] (2x8) -> C[:,0:8] (2x8)
    # Split into 4 blocks of 2 columns each, apply Strassen 2x2x2 to each
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1

        # Local indices:
        # A: a00=A[0,1], a01=A[0,2], a10=A[1,1], a11=A[1,2]
        # B: b00=B[1,k0], b01=B[1,k1], b10=B[2,k0], b11=B[2,k1]
        # C: c00=C[0,k0], c01=C[0,k1], c10=C[1,k0], c11=C[1,k1]

        # Global indices:
        u_a00 = 0 * m + 1  # A[0,1]
        u_a01 = 0 * m + 2  # A[0,2]
        u_a10 = 1 * m + 1  # A[1,1]
        u_a11 = 1 * m + 2  # A[1,2]

        v_b00 = 1 * p + k0
        v_b01 = 1 * p + k1
        v_b10 = 2 * p + k0
        v_b11 = 2 * p + k1

        w_c00 = k0 * n + 0
        w_c01 = k1 * n + 0
        w_c10 = k0 * n + 1
        w_c11 = k1 * n + 1

        # Strassen 7 multiplications:
        # M1 = (a00 + a11) * (b00 + b11)
        U[u_a00][col] = 1; U[u_a11][col] = 1
        V[v_b00][col] = 1; V[v_b11][col] = 1
        W[w_c00][col] = 1; W[w_c11][col] = 1
        col += 1

        # M2 = (a10 + a11) * b00
        U[u_a10][col] = 1; U[u_a11][col] = 1
        V[v_b00][col] = 1
        W[w_c10][col] = 1; W[w_c11][col] = -1
        col += 1

        # M3 = a00 * (b01 - b11)
        U[u_a00][col] = 1
        V[v_b01][col] = 1; V[v_b11][col] = -1
        W[w_c01][col] = 1; W[w_c11][col] = 1
        col += 1

        # M4 = a11 * (b10 - b00)
        U[u_a11][col] = 1
        V[v_b10][col] = 1; V[v_b00][col] = -1
        W[w_c00][col] = 1; W[w_c10][col] = 1
        col += 1

        # M5 = (a00 + a01) * b11
        U[u_a00][col] = 1; U[u_a01][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = -1; W[w_c01][col] = 1
        col += 1

        # M6 = (a10 - a00) * (b00 + b01)
        U[u_a10][col] = 1; U[u_a00][col] = -1
        V[v_b00][col] = 1; V[v_b01][col] = 1
        W[w_c11][col] = 1
        col += 1

        # M7 = (a01 - a11) * (b10 + b11)
        U[u_a01][col] = 1; U[u_a11][col] = -1
        V[v_b10][col] = 1; V[v_b11][col] = 1
        W[w_c00][col] = 1
        col += 1

    # Verify the rank-44 construction
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 48
    rank_fallback = n * m * p
    U_f = [[0] * rank_fallback for _ in range(n * m)]
    V_f = [[0] * rank_fallback for _ in range(m * p)]
    W_f = [[0] * rank_fallback for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_f[i * m + j][col] = 1
                V_f[j * p + k][col] = 1
                W_f[k * n + i][col] = 1
                col += 1
    return {"U": U_f, "V": V_f, "W": W_f}
# EVOLVE-BLOCK-END
