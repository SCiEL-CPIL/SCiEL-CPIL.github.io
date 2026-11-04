"""Matrix multiplication 4x5x6: search for rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 142.0

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

    def verify(result, atol=1e-8):
        U = np.array(result["U"], dtype=np.float64)
        V = np.array(result["V"], dtype=np.float64)
        W = np.array(result["W"], dtype=np.float64)
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
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
                target_indices.append((a, b, c))

    # Precompute flattened target in different modes
    # Mode-0 flattening: for each a, target[a, b, c] -> (N, M*P)
    target_a = target.reshape(N, M * P)
    # Mode-1 flattening: for each b, target[a, b, c] -> (M, N*P)
    target_b = target.transpose(1, 0, 2).reshape(M, N * P)
    # Mode-2 flattening: for each c, target[a, b, c] -> (P, N*M)
    target_c = target.transpose(2, 0, 1).reshape(P, N * M)

    best = schoolbook()
    best_rank = n * m * p

    def als_decompose_fast(target, target_a, target_b, target_c, rank, max_iter=60, tol=1e-10):
        """Fast ALS decomposition using explicit matrix products."""
        np.random.seed(12345)
        U = np.random.randn(N, rank) * 0.5 + 0.5
        V = np.random.randn(M, rank) * 0.5 + 0.5
        W = np.random.randn(P, rank) * 0.5 + 0.5
        
        prev_err = np.inf
        for iteration in range(max_iter):
            if time_remaining() < 15:
                break
            
            # Update W: fix U, V
            # For each c, W[c, r] such that sum_r W[c,r] * (sum_a U[a,r] * target_b...)
            # Actually: W[c, r] * sum_a,b U[a,r]*V[b,r] should match target_c[c, a*M+b]
            # W @ (U^T * V elementwise summed over a,b) ... 
            # Let me think: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed c: sum_r W[c,r] * (U^T V)[a,b,r] ... no
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # Fix U,V: for each c, we need W[c,:] such that
            # sum_r W[c,r] * (U^T V)[r, a, b] ... 
            # Actually (U^T V) doesn't make sense directly.
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # = sum_r W[c,r] * (U[a,r] V[b,r])
            # So for fixed c: target_c[c, a*M+b] = sum_r W[c,r] * (U[a,r]*V[b,r])
            # Let M_c[a*M+b, r] = U[a,r] * V[b,r]
            # Then target_c[c, :] = W[c, :] @ M_c.T
            # So W[c, :] = (target_c[c, :] @ M_c) @ inv(M_c.T @ M_c) ... or lstsq
            
            M_c = (U[:, :, None] * V[None, :, :]).reshape(N * M, rank)  # (N*M, rank)
            # W[c, r] = lstsq(M_c, target_c[c, :])
            try:
                W_new = np.linalg.lstsq(M_c, target_c.T, rcond=None)[0].T  # (P, rank)
                W = W_new
            except:
                pass
            
            # Update U: fix V, W
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed a: target_a[a, b*P+c] = sum_r U[a,r] * (V[b,r] W[c,r])
            # M_U[b*P+c, r] = V[b,r] * W[c,r]
            M_U = (V[:, :, None] * W[None, :, :]).reshape(M * P, rank)  # (M*P, rank)
            try:
                U_new = np.linalg.lstsq(M_U, target_a.T, rcond=None)[0].T  # (N, rank)
                U = U_new
            except:
                pass
            
            # Update V: fix U, W
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed b: target_b[b, a*P+c] = sum_r V[b,r] * (U[a,r] W[c,r])
            M_V = (U[:, :, None] * W[None, :, :]).reshape(N * P, rank)  # (N*P, rank)
            try:
                V_new = np.linalg.lstsq(M_V, target_b.T, rcond=None)[0].T  # (M, rank)
                V = V_new
            except:
                pass
            
            # Compute error
            T = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T - target)
            if abs(prev_err - err) < tol and err < 1e-8:
                break
            prev_err = err
        
        return U, V, W

    def try_rank(r, num_trials=8):
        """Try to find a decomposition of rank r."""
        if time_remaining() < 20:
            return None
        for trial in range(num_trials):
            if time_remaining() < 15:
                break
            np.random.seed(1000 + trial * 7919)
            U, V, W = als_decompose_fast(target, target_a, target_b, target_c, r, max_iter=80)
            result = {"U": U, "V": V, "W": W}
            if verify(result, atol=1e-7):
                return result
        return None

    # The schoolbook rank is 120. Let's search downward.
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # But practical bounds are much higher. Let's try from 119 down.
    # Focus on a range that's likely to have solutions.
    
    # First, try a few ranks near the schoolbook to see if we can improve
    search_ranks = list(range(119, 90, -1))
    
    for r in search_ranks:
        if time_remaining() < 25:
            break
        result = try_rank(r, num_trials=6)
        if result is not None:
            best = result
            best_rank = r
            print(f"Found decomposition with rank {r}")
            # Continue searching for even lower ranks
            continue

    # If we found something, try to go lower
    if best_rank < 120 and time_remaining() > 40:
        for r in range(best_rank - 1, max(best_rank - 30, 50), -1):
            if time_remaining() < 20:
                break
            result = try_rank(r, num_trials=4)
            if result is not None:
                best = result
                best_rank = r
                print(f"Found decomposition with rank {r}")

    # Final verification
    if not verify(best, atol=1e-7):
        best = schoolbook()
        best_rank = n * m * p

    # Round to nearest half-integer if close
    U_out = np.array(best["U"])
    V_out = np.array(best["V"])
    W_out = np.array(best["W"])
    
    U_rounded = np.round(U_out * 2) / 2
    V_rounded = np.round(V_out * 2) / 2
    W_rounded = np.round(W_out * 2) / 2
    
    result_rounded = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
    if verify(result_rounded, atol=1e-6):
        best = result_rounded

    return {"U": best["U"].tolist(), "V": best["V"].tolist(), "W": best["W"].tolist()}
# EVOLVE-BLOCK-END
