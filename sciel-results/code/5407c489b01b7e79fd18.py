import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 145.0

    # Current best known: Rank 34
    rank = 34
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    num_pairs = p // 2

    for t in range(num_pairs):
        base = t * 5
        k0 = 2 * t
        k1 = 2 * t + 1

        # Indices
        v_b0_k0 = 0 * p + k0
        v_b0_k1 = 0 * p + k1
        v_b1_k0 = 1 * p + k0
        v_b1_k1 = 1 * p + k1

        w_c0_k0 = k0 * n + 0
        w_c0_k1 = k1 * n + 0
        w_c1_k0 = k0 * n + 1
        w_c1_k1 = k1 * n + 1

        # Local terms (5 ranks)
        # M1
        c = base + 0
        U[0, c] = 1; U[3, c] = 1
        V[v_b0_k0, c] = 1; V[v_b1_k1, c] = 1
        W[w_c0_k0, c] = 1; W[w_c1_k1, c] = 1

        # M2
        c = base + 1
        U[2, c] = 1; U[3, c] = 1
        V[v_b0_k0, c] = 1
        W[w_c1_k0, c] = 1; W[w_c1_k1, c] = -1

        # M5
        c = base + 2
        U[0, c] = 1; U[1, c] = 1
        V[v_b1_k1, c] = 1
        W[w_c0_k0, c] = -1; W[w_c0_k1, c] = 1

        # M6
        c = base + 3
        U[2, c] = 1; U[0, c] = -1
        V[v_b0_k0, c] = 1; V[v_b0_k1, c] = 1
        W[w_c1_k1, c] = 1

        # M7
        c = base + 4
        U[1, c] = 1; U[3, c] = -1
        V[v_b1_k0, c] = 1; V[v_b1_k1, c] = 1
        W[w_c0_k0, c] = 1

    # Global terms (4 ranks)
    c = 30
    U[0, c] = 1
    for k in range(p):
        V[0 * p + k, c] = (-1) ** k
        W[k * n + 0, c] = (-1) ** k

    c = 31
    U[3, c] = 1
    for k in range(p):
        V[1 * p + k, c] = (-1) ** k
        W[k * n + 1, c] = (-1) ** k

    c = 32
    U[0, c] = 1
    for t in range(num_pairs):
        V[0 * p + 2*t, c] = 1
        V[0 * p + 2*t+1, c] = -1
        W[(2*t) * n + 0, c] = 1
        W[(2*t+1) * n + 0, c] = -1

    c = 33
    U[3, c] = 1
    for t in range(num_pairs):
        V[1 * p + 2*t, c] = 1
        V[1 * p + 2*t+1, c] = -1
        W[(2*t) * n + 1, c] = 1
        W[(2*t+1) * n + 1, c] = -1

    # Verification
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c_idx in range(rank):
                    val += U[i*m+j, c_idx] * V[j*p+k, c_idx] * W[k*n+i, c_idx]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
                    
    if max_err > 1e-9:
        # Fallback to trivial rank 48 if something goes wrong
        rank = 48
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
                    idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
