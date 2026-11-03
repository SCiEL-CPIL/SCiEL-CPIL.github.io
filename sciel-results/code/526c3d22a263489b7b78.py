"""Matrix multiplication 4x5x6: search for rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 135.0  # 15s safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def time_remaining():
        return DEADLINE - elapsed()

    n, m, p = DIMENSIONS
    N = n * m  # 20
    M = m * p  # 30
    P = p * n  # 24

    def schoolbook():
        rank = n * m * p
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def to_lists(mat):
        return mat.tolist()

    def verify(result, atol=1e-6):
        U = np.array(result["U"])
        V = np.array(result["V"])
        W = np.array(result["W"])
        if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
            return False
        if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
            return False
        if U.shape[0] != N or V.shape[0] != M or W.shape[0] != P:
            return False
        T = np.einsum('ar,br,cr->abc', U, V, W)
        target = np.zeros((N, M, P))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1.0
        return np.allclose(T, target, atol=atol)

    # Build target tensor once
    target = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    # Start with schoolbook
    best = schoolbook()
    best_rank = n * m * p

    def als_decompose(target, rank, max_iter=50, tol=1e-8):
        """ALS decomposition with better initialization and convergence check."""
        # Initialize with small random values
        U = np.random.randn(N, rank) * 0.5
        V = np.random.randn(M, rank) * 0.5
        W = np.random.randn(P, rank) * 0.5

        prev_err = np.inf
        for iteration in range(max_iter):
            if time_remaining() < 15:
                break

            # Update U: fix V, W
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed V,W: minimize ||target - sum_r U[:,r] (V[:,r] ⊗ W[:,r])^T||_F
            # Build H_U: H_U[r, b*P+c] = V[b,r] * W[c,r]
            H_U = np.array([np.kron(V[:, r], W[:, r]) for r in range(rank)]).T  # (M*P, rank)
            target_flat = target.reshape(N, M * P)
            try:
                U_new, _, _, _ = np.linalg.lstsq(H_U, target_flat.T, rcond=None)
                U = U_new.T  # (N, rank)
            except:
                pass

            # Update V: fix U, W
            # H_V[r, a*P+c] = U[a,r] * W[c,r]
            H_V = np.array([np.kron(U[:, r], W[:, r]) for r in range(rank)]).T  # (N*P, rank)
            target_v = target.transpose(1, 0, 2).reshape(M, N * P)
            try:
                V_new, _, _, _ = np.linalg.lstsq(H_V, target_v.T, rcond=None)
                V = V_new.T  # (M, rank)
            except:
                pass

            # Update W: fix U, V
            # H_W[r, a*M+b] = U[a,r] * V[b,r]
            H_W = np.array([np.kron(U[:, r], V[:, r]) for r in range(rank)]).T  # (N*M, rank)
            target_w = target.transpose(2, 0, 1).reshape(P, N * M)
            try:
                W_new, _, _, _ = np.linalg.lstsq(H_W, target_w.T, rcond=None)
                W = W_new.T  # (P, rank)
            except:
                pass

            # Check convergence
            T = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T - target)
            if abs(prev_err - err) < tol:
                break
            prev_err = err

        return U, V, W

    def try_rank(r, num_trials=5):
        """Try to find a decomposition of rank r."""
        if time_remaining() < 25:
            return None
        for trial in range(num_trials):
            if time_remaining() < 20:
                break
            U, V, W = als_decompose(target, r, max_iter=40)
            result = {"U": U, "V": V, "W": W}
            if verify(result):
                return result
        return None

    # Try ranks below schoolbook, starting from 119 down
    for r in range(best_rank - 1, max(best_rank - 25, 95), -1):
        if time_remaining() < 35:
            break
        result = try_rank(r, num_trials=3)
        if result is not None:
            best = result
            best_rank = r
            print(f"Found decomposition with rank {r}")
            continue

    # Final verification with tighter tolerance
    if not verify(best, atol=1e-5):
        best = schoolbook()
        best_rank = n * m * p

    return {"U": to_lists(best["U"]), "V": to_lists(best["V"]), "W": to_lists(best["W"])}
# EVOLVE-BLOCK-END
