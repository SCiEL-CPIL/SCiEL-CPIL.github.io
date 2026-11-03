"""Matrix multiplication 4x5x6: search for rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # 10s safety margin before 150s kill

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

    def verify(result):
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
        return np.allclose(T, target, atol=1e-6)

    # Build target tensor once
    target = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    # Start with schoolbook
    best = schoolbook()
    best_rank = n * m * p

    def als_step(U, V, W, target, max_iter=50):
        """Alternating Least Squares to minimize ||target - sum_r U_r V_r W_r||_F"""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = U.shape[1]
        
        for _ in range(max_iter):
            if time_remaining() < 5:
                break
            
            # Update U: fix V, W. T[a,:,:] = sum_r U[a,r] (V[:,r] * W[:,r])
            # H_U[r, b*P+c] = V[b,r] * W[c,r] = kron(V[:,r], W[:,r])
            H_U = np.zeros((r, M * P))
            for rr in range(r):
                H_U[rr, :] = np.kron(V[:, rr], W[:, rr])
            target_reshaped = target.reshape(N, M * P)
            try:
                U_new, _, _, _ = np.linalg.lstsq(H_U.T, target_reshaped, rcond=None)
                U = U_new
            except:
                pass

            # Update V: fix U, W. T[:,b,:] = sum_r V[b,r] (U[:,r] * W[:,r])
            # H_V[r, a*P+c] = U[a,r] * W[c,r] = kron(U[:,r], W[:,r])
            H_V = np.zeros((r, N * P))
            for rr in range(r):
                H_V[rr, :] = np.kron(U[:, rr], W[:, rr])
            target_v_reshaped = target.transpose(1, 0, 2).reshape(M, N * P)
            try:
                V_new, _, _, _ = np.linalg.lstsq(H_V.T, target_v_reshaped, rcond=None)
                V = V_new
            except:
                pass

            # Update W: fix U, V. T[:,:,c] = sum_r W[c,r] (U[:,r] * V[:,r])
            # H_W[r, a*M+b] = U[a,r] * V[b,r] = kron(U[:,r], V[:,r])
            H_W = np.zeros((r, N * M))
            for rr in range(r):
                H_W[rr, :] = np.kron(U[:, rr], V[:, rr])
            target_w_reshaped = target.transpose(2, 0, 1).reshape(P, N * M)
            try:
                W_new, _, _, _ = np.linalg.lstsq(H_W.T, target_w_reshaped, rcond=None)
                W = W_new
            except:
                pass

        return U, V, W

    def try_rank(r, num_trials=5):
        """Try to find a decomposition of rank r using ALS with random init."""
        if time_remaining() < 20:
            return None
        for trial in range(num_trials):
            if time_remaining() < 15:
                break
            
            # Random initialization with small scale
            scale = 0.1
            U = np.random.randn(N, r) * scale
            V = np.random.randn(M, r) * scale
            W = np.random.randn(P, r) * scale
            
            # Run ALS
            U, V, W = als_step(U, V, W, target, max_iter=40)
            
            # Check residual
            T = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(T - target)
            
            if residual < 1e-5:
                return {"U": U, "V": V, "W": W}
        
        return None

    # Try ranks below schoolbook
    # Start from 119 down to 100 (or lower if time permits)
    for r in range(best_rank - 1, max(90, best_rank - 30), -1):
        if time_remaining() < 30:
            break
        result = try_rank(r, num_trials=3)
        if result is not None:
            best = result
            best_rank = r
            print(f"Found decomposition with rank {r}")
            # Try to go even lower immediately
            continue

    # Final verification
    if not verify(best):
        # Fall back to schoolbook
        best = schoolbook()
        best_rank = n * m * p

    return {"U": to_lists(best["U"]), "V": to_lists(best["V"]), "W": to_lists(best["W"])}
# EVOLVE-BLOCK-END
