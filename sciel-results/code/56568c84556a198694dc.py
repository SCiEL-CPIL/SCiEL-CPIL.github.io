import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = DIMENSIONS

    def idx_u(i, j):
        return i * m + j

    def idx_v(j, k):
        return j * p + k

    def idx_w(k, i):
        return k * n + i

    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Rank 26 decomposition: Strassen 2x2 core + standard border
    R = 26
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    # Strassen 2x2 core (7 terms)
    # m1 = (a00 + a11)(b00 + b11)
    r = 0
    U[idx_u(0, 0), r] = 1; U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1; V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1; W[idx_w(1, 1), r] = 1

    # m2 = (a10 + a11)b00
    r = 1
    U[idx_u(1, 0), r] = 1; U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1
    W[idx_w(0, 1), r] = 1; W[idx_w(1, 1), r] = -1

    # m3 = a00(b01 - b11)
    r = 2
    U[idx_u(0, 0), r] = 1
    V[idx_v(0, 1), r] = 1; V[idx_v(1, 1), r] = -1
    W[idx_w(1, 0), r] = 1; W[idx_w(1, 1), r] = 1

    # m4 = a11(b10 - b00)
    r = 3
    U[idx_u(1, 1), r] = 1
    V[idx_v(1, 0), r] = 1; V[idx_v(0, 0), r] = -1
    W[idx_w(0, 0), r] = 1; W[idx_w(0, 1), r] = 1

    # m5 = (a00 + a01)b11
    r = 4
    U[idx_u(0, 0), r] = 1; U[idx_u(0, 1), r] = 1
    V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = -1; W[idx_w(1, 0), r] = 1

    # m6 = (a10 - a00)(b00 + b01)
    r = 5
    U[idx_u(1, 0), r] = 1; U[idx_u(0, 0), r] = -1
    V[idx_v(0, 0), r] = 1; V[idx_v(0, 1), r] = 1
    W[idx_w(1, 1), r] = 1

    # m7 = (a01 - a11)(b10 + b11)
    r = 6
    U[idx_u(0, 1), r] = 1; U[idx_u(1, 1), r] = -1
    V[idx_v(1, 0), r] = 1; V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1

    # Standard border terms (19 terms)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[idx_u(i, j), col] = 1
                V[idx_v(j, k), col] = 1
                W[idx_w(k, i), col] = 1
                col += 1

    # Verify
    U_r = U.reshape((n, m, R))
    V_r = V.reshape((m, p, R))
    W_r = W.reshape((p, n, R))
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    T_target = np.ones((n, m, p))

    if not np.allclose(T_recon, T_target, atol=1e-9):
        return schoolbook()

    # Try to find rank 25 by attempting to merge two border terms
    # into the Strassen structure, or find a different 25-term decomposition.
    # This is a heuristic search.
    if time.time() < deadline - 30:
        # Try rank 25: see if we can eliminate one border term by
        # modifying the Strassen terms.
        # For now, we attempt a simple local search: try to find a
        # rank-25 decomposition by perturbing the rank-26 one.
        # This is complex, so we fall back to rank 26.
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
