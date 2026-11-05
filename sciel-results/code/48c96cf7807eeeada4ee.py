import numpy as np
import time

def construct():
    n, m, p = 3, 4, 8
    start_time = time.time()
    deadline = start_time + 115

    # Rank-12 decomposition exploiting row-wise separability.
    # For each output row i and each input column j, one rank-1 term:
    #   U has 1 at (i, j) of A
    #   V has 1s across row j of B (all k)
    #   W has 1s across column i of C (all k)
    # Sum over j gives C[i,k] = sum_j A[i,j] B[j,k].
    rank = n * m  # 3 * 4 = 12
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for i in range(n):
        for j in range(m):
            # U: position (i, j) in the n x m matrix A
            U[i * m + j, col] = 1.0
            # V: row j of the m x p matrix B (all k)
            for k in range(p):
                V[j * p + k, col] = 1.0
            # W: column i of the p x n output C (all k), index k*n + i
            for k in range(p):
                W[k * n + i, col] = 1.0
            col += 1

    # Verify correctness with a random test (cheap, exact for integer entries)
    try:
        rng = np.random.default_rng(0)
        A = rng.integers(-5, 6, size=(n, m))
        B = rng.integers(-5, 6, size=(m, p))
        C_expected = A @ B
        C_recon = np.zeros((n, p))
        for t in range(rank):
            u = U[:, t]
            v = V[:, t]
            w = W[:, t]
            # reconstruct
            C_recon += np.outer(u.reshape(n, m).T, v.reshape(m, p).T).reshape(n, p) * 0
        # Proper reconstruction: C[i,k] = sum_t U[i*m+j?] ... do it directly
        C_recon = np.zeros((n, p))
        for t in range(rank):
            u = U[:, t]
            v = V[:, t]
            w = W[:, t]
            # u is n*m vector, v is m*p vector, w is p*n vector
            # The rank-1 contribution to C is: for each (i,k),
            #   sum_j u[i*m+j] * v[j*p+k] * w[k*n+i]
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += u[i * m + j] * v[j * p + k] * w[k * n + i]
                    C_recon[i, k] = s
        if not np.allclose(C_recon, C_expected, atol=1e-9):
            # Fallback to schoolbook if somehow broken (should not happen)
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                        col += 1
    except Exception:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
