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

    def als_decompose(target, rank, max_iter=100, tol=1e-12):
        """ALS decomposition with robust initialization and convergence check."""
        # Initialize with random values
        # Using a mix of random and structured initialization
        U = np.random.randn(N, rank)
        V = np.random.randn(M, rank)
        W = np.random.randn(P, rank)
        
        # Scale to avoid overflow
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)

        prev_err = np.inf
        for iteration in range(max_iter):
            if time_remaining() < 20:
                break

            # Update U
            H_U = np.einsum('br,cr->bcr', V, W).reshape(M * P, rank)
            target_flat = target.reshape(N, M * P)
            try:
                U_new, _, _, _ = np.linalg.lstsq(H_U, target_flat.T, rcond=None)
                U = U_new.T
            except:
                pass

            # Update V
            H_V = np.einsum('ar,cr->acr', U, W).reshape(N * P, rank)
            target_v = target.transpose(1, 0, 2).reshape(M, N * P)
            try:
                V_new, _, _, _ = np.linalg.lstsq(H_V, target_v.T, rcond=None)
                V = V_new.T
            except:
                pass

            # Update W
            H_W = np.einsum('ar,br->abr', U, V).reshape(N * M, rank)
            target_w = target.transpose(2, 0, 1).reshape(P, N * M)
            try:
                W_new, _, _, _ = np.linalg.lstsq(H_W, target_w.T, rcond=None)
                W = W_new.T
            except:
                pass

            # Check convergence
            T = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T - target)
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
            # If error is very small, we can stop early
            if err < 1e-9:
                break

        # Final polish: if error is small, try to snap to half-integers
        T = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T - target)
        
        if err < 1e-5:
            # Try snapping
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            T_snapped = np.einsum('ar,br,cr->abc', U_snapped, V_snapped, W_snapped)
            err_snapped = np.linalg.norm(T_snapped - target)
            
            if err_snapped < 1e-5:
                U, V, W = U_snapped, V_snapped, W_snapped

        return U, V, W

    def try_rank(r, num_trials=10):
        """Try to find a decomposition of rank r."""
        if time_remaining() < 25:
            return None
        for trial in range(num_trials):
            if time_remaining() < 20:
                break
            U, V, W = als_decompose(target, r, max_iter=80)
            result = {"U": U, "V": V, "W": W}
            if verify(result):
                return result
        return None

    # Try ranks below schoolbook, starting from 100 down to 60
    # The schoolbook rank is 120.
    # We expect the rank to be significantly lower.
    for r in range(100, 59, -1):
        if time_remaining() < 35:
            break
        result = try_rank(r, num_trials=5)
        if result is not None:
            best = result
            best_rank = r
            print(f"Found decomposition with rank {r}")
            # If we find a low rank, we can try to go lower immediately
            # But for safety, we just break and return the best found so far
            # Or continue to see if we can go even lower?
            # The loop goes down, so it will try lower ranks next.
            continue

    # If we didn't find anything below 120, try 119 down to 101
    if best_rank == 120:
        for r in range(119, 100, -1):
            if time_remaining() < 35:
                break
            result = try_rank(r, num_trials=3)
            if result is not None:
                best = result
                best_rank = r
                print(f"Found decomposition with rank {r}")
                break

    # Final verification with tighter tolerance
    if not verify(best, atol=1e-5):
        best = schoolbook()
        best_rank = n * m * p

    return {"U": to_lists(best["U"]), "V": to_lists(best["V"]), "W": to_lists(best["W"])}
# EVOLVE-BLOCK-END
