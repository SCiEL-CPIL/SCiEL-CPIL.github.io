"""Matrix multiplication 4x5x6: search for rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

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
        return np.allclose(T, target, atol=1e-7)

    # Build target tensor once
    target = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    # Start with schoolbook
    best = schoolbook()
    best_rank = n * m * p

    # Try to find a lower rank decomposition using ALS (Alternating Least Squares)
    # We'll try several random initializations and different target ranks.
    
    # Known bounds: for 4x5x6, schoolbook gives 120.
    # Let's try ranks below 120.
    # Actually, the minimum rank might be exactly 120 or slightly less.
    # Let's try a few target ranks: 119, 118, ..., down to maybe 100.
    
    # ALS approach:
    # T = sum_r U[:, r] (V[:, r] W[:, r])^T  -- wait, that's not right.
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    # For fixed V and W, we want to find U such that T ≈ sum_r U[:,r] (V[:,r] ⊗ W[:,r])^T
    # This is a linear least squares problem for each column of U.
    # Let's reshape T to (N, M*P) where T[a, b*M+c] = T[a,b,c]
    # And for each r, let G_r = V[:,r] ⊗ W[:,r]  (length M*P)
    # Then T = sum_r U[:,r] G_r^T
    # So U = T @ G @ pinv(G @ G^T)  -- or more carefully using lstsq.
    
    # Actually, let's use a cleaner formulation.
    # Fix V and W. For each r, define g_r = V[:,r] * W[:,r]  (element-wise, length M... no, that's wrong)
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Let's define for each r, a vector h_r of length M*P where h_r[b*P+c] = V[b,r] W[c,r]
    # Then T = sum_r U[:,r] h_r^T
    # So we have T (N x M*P) = U (N x r) * H^T (r x M*P) where H[r, b*P+c] = V[b,r] W[c,r]
    # To solve for U: U = T @ pinv(H)  -- but this doesn't work directly because H is r x (M*P)
    # We want U such that T = U H^T
    # This is a matrix factorization. Use least squares: U = T @ H^T @ pinv(H H^T)
    # Or more simply, use np.linalg.lstsq
    
    def als_step_fix_U(U, V, W, target, max_iter=20):
        """Fix U, update V and W alternately."""
        V = V.copy()
        W = W.copy()
        for _ in range(max_iter):
            if time_remaining() < 10:
                break
            # Fix V and W, update U
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For each a, T[a,:,:] = sum_r U[a,r] (V[:,r] W[:,r]^T)  -- outer product
            # Let's build the Gram matrix approach
            # For U: minimize ||target - sum_r U[:,r] (V[:,r] ⊙ W[:,r])||_F
            # where ⊙ is the Hadamard product reshaped appropriately.
            
            # Build H_U: for each r, h_r of length M*P: h_r[b*P+c] = V[b,r] * W[c,r]
            H_U = np.zeros((V.shape[1], M * P))
            for r in range(V.shape[1]):
                H_U[r, :] = (V[:, r] * W[:, r]).reshape(-1)
            # target_reshaped: (N, M*P)
            target_reshaped = target.reshape(N, M * P)
            # U = argmin ||target_reshaped - U @ H_U.T||_F
            # This is a linear least squares: U @ H_U.T ≈ target_reshaped
            # U = target_reshaped @ H_U.T @ pinv(H_U @ H_U.T)
            try:
                U_new, _, _, _ = np.linalg.lstsq(H_U.T, target_reshaped, rcond=None)
                U = U_new
            except:
                pass
            
            # Fix U and W, update V
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For each b: T[:,:,b] = sum_r V[b,r] (U[:,r] W[:,r]^T)
            # Let's build H_V: for each r, h_r of length N*P: h_r[a*P+c] = U[a,r] * W[c,r]
            H_V = np.zeros((V.shape[1], N * P))
            for r in range(V.shape[1]):
                H_V[r, :] = (U[:, r] * W[:, r]).reshape(-1)
            target_v_reshaped = target.transpose(1, 0, 2).reshape(M, N * P)
            try:
                V_new, _, _, _ = np.linalg.lstsq(H_V.T, target_v_reshaped, rcond=None)
                V = V_new
            except:
                pass
            
            # Fix U and V, update W
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For each c: T[:,:,c] = sum_r W[c,r] (U[:,r] V[:,r]^T)
            H_W = np.zeros((W.shape[1], N * M))
            for r in range(W.shape[1]):
                H_W[r, :] = (U[:, r] * V[:, r]).reshape(-1)
            target_w_reshaped = target.transpose(2, 0, 1).reshape(P, N * M)
            try:
                W_new, _, _, _ = np.linalg.lstsq(H_W.T, target_w_reshaped, rcond=None)
                W = W_new
            except:
                pass
        
        return U, V, W

    def try_rank(r, num_trials=3):
        """Try to find a decomposition of rank r."""
        if time_remaining() < 20:
            return None
        for trial in range(num_trials):
            if time_remaining() < 15:
                break
            # Random initialization
            U = np.random.randn(N, r) * 0.1
            V = np.random.randn(M, r) * 0.1
            W = np.random.randn(P, r) * 0.1
            
            # Run ALS
            U, V, W = als_step_fix_U(U, V, W, target, max_iter=30)
            
            # Check if we found a valid decomposition
            result = {"U": U, "V": V, "W": W}
            if verify(result):
                return result
        return None

    # Try ranks below schoolbook
    for r in range(best_rank - 1, best_rank - 20, -1):
        if time_remaining() < 30:
            break
        result = try_rank(r, num_trials=2)
        if result is not None:
            best = result
            best_rank = r
            print(f"Found decomposition with rank {r}")
            # Try to go even lower
            continue
    
    # Final verification
    if not verify(best):
        # Fall back to schoolbook
        best = schoolbook()
        best_rank = n * m * p

    return {"U": to_lists(best["U"]), "V": to_lists(best["V"]), "W": to_lists(best["W"])}
# EVOLVE-BLOCK-END
