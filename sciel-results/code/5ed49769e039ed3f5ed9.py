import numpy as np
import time

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    start_time = time.time()
    deadline = 100  # seconds, leave margin for 120s limit

    # Build the full tensor T of shape (4, 32, 32)
    T_full = np.zeros((4, 32, 32))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_full[a, b, c] = 1

    # 2x2x2 sub-tensor
    T_222 = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a = i * 2 + j
                b = j * 2 + k
                c = k * 2 + i
                T_222[a, b, c] = 1

    # Find rank-7 decomposition of 2x2x2 via alternating least squares
    R = 7
    best_U2 = None
    best_V2 = None
    best_W2 = None
    best_err = float('inf')

    for trial in range(5):
        if time.time() - start_time > deadline - 5:
            break
        np.random.seed(42 + trial)
        U2 = np.random.randn(4, R) * 0.5
        V2 = np.random.randn(4, R) * 0.5
        W2 = np.random.randn(4, R) * 0.5

        for iteration in range(300):
            if time.time() - start_time > deadline - 5:
                break

            # Update U2
            M = np.zeros((16, R))
            for r in range(R):
                M[:, r] = np.outer(V2[:, r], W2[:, r]).flatten()
            for a in range(4):
                U2[a, :] = np.linalg.lstsq(M, T_222[a, :, :].flatten(), rcond=None)[0]

            # Update V2
            M = np.zeros((16, R))
            for r in range(R):
                M[:, r] = np.outer(U2[:, r], W2[:, r]).flatten()
            for b in range(4):
                V2[b, :] = np.linalg.lstsq(M, T_222[:, b, :].flatten(), rcond=None)[0]

            # Update W2
            M = np.zeros((16, R))
            for r in range(R):
                M[:, r] = np.outer(U2[:, r], V2[:, r]).flatten()
            for c in range(4):
                W2[c, :] = np.linalg.lstsq(M, T_222[:, :, c].flatten(), rcond=None)[0]

            # Check error
            T_approx = np.zeros((4, 4, 4))
            for r in range(R):
                T_approx += np.outer(np.outer(U2[:, r], V2[:, r]), W2[:, r])
            err = np.linalg.norm(T_approx - T_222)
            if err < best_err:
                best_err = err
                best_U2 = U2.copy()
                best_V2 = V2.copy()
                best_W2 = W2.copy()
            if err < 1e-12:
                break

        if best_err < 1e-10:
            break

    if best_err < 1e-8:
        # Extend to 2x2x16: 8 blocks of 2 columns
        rank = 56
        U = np.zeros((4, rank))
        V = np.zeros((32, rank))
        W = np.zeros((32, rank))

        for g in range(8):
            r_off = g * 7
            for r in range(7):
                U[:, r_off + r] = best_U2[:, r]
                for j in range(2):
                    for k in range(2):
                        b_loc = j * 2 + k
                        b_glb = j * 16 + (2 * g + k)
                        V[b_glb, r_off + r] = best_V2[b_loc, r]
                for i in range(2):
                    for k in range(2):
                        c_loc = k * 2 + i
                        c_glb = (2 * g + k) * 2 + i
                        W[c_glb, r_off + r] = best_W2[c_loc, r]

        # Verify
        T_approx = np.zeros((4, 32, 32))
        for r in range(rank):
            T_approx += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])

        if np.linalg.norm(T_approx - T_full) < 1e-8:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank 64
    rank = 64
    U = np.zeros((4, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
