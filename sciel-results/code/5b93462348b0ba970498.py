import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline, 20s buffer

    # Build target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def compute_error(U, V, W):
        R = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(R - T))

    # Schoolbook decomposition (rank = n*m*p = 112)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1

    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = rank_sb

    def als_optimize(T, rank, max_iter=150, tol=1e-7, deadline=None):
        n1, n2, n3 = T.shape
        np.random.seed(42)  # For reproducibility in trials
        U = np.random.randn(n1, rank) * 0.5
        V = np.random.randn(n2, rank) * 0.5
        W = np.random.randn(n3, rank) * 0.5

        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break

            # Update U
            M = np.einsum('br,cr->bcr', V, W).reshape(n2*n3, rank)
            T_flat = T.reshape(n1, n2*n3)
            U_new, _, _, _ = np.linalg.lstsq(M.T, T_flat.T, rcond=None)
            U = U_new.T

            # Update V
            N = np.einsum('ar,cr->acr', U, W).reshape(n1*n3, rank)
            T_flat_v = T.transpose(1, 0, 2).reshape(n2, n1*n3)
            V_new, _, _, _ = np.linalg.lstsq(N.T, T_flat_v.T, rcond=None)
            V = V_new.T

            # Update W
            P = np.einsum('ar,br->abr', U, V).reshape(n1*n2, rank)
            T_flat_w = T.transpose(2, 0, 1).reshape(n3, n1*n2)
            W_new, _, _, _ = np.linalg.lstsq(P.T, T_flat_w.T, rcond=None)
            W = W_new.T

        return U, V, W, compute_error(U, V, W)

    # Try to find lower-rank decompositions
    for rank in [80, 70, 60, 50, 45]:
        if time.time() > deadline:
            break
        found = False
        for trial in range(8):
            if time.time() > deadline:
                break
            # Vary random seed for diversity
            np.random.seed(42 + trial)
            U, V, W, err = als_optimize(T, rank, max_iter=80, deadline=deadline)
            if err < 1e-6 and rank < best_rank:
                best_rank = rank
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                found = True
                break

    return best_result
# EVOLVE-BLOCK-END
