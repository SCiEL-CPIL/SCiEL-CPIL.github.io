"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = 3, 3, 7
    rank_sb = n * m * p  # 63
    
    # Schoolbook decomposition - always valid
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_sb
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    def verify(U, V, W, tol=1e-10):
        T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_approx, T, atol=tol)
    
    def get_residual(U, V, W):
        T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.linalg.norm(T_approx - T)
    
    # Try ranks with ALS, using better initialization and more attempts
    for target_rank in range(62, 45, -1):
        if time.time() > deadline - 25.0:
            break
        
        # More attempts for higher ranks, fewer for lower
        max_attempts = 4 if target_rank >= 58 else 3
        for attempt in range(max_attempts):
            if time.time() > deadline - 25.0:
                break
            
            np.random.seed(2000 + target_rank * 100 + attempt)
            
            # Structured initialization based on schoolbook
            if target_rank >= rank_sb:
                U = U_sb[:, :target_rank].copy()
                V = V_sb[:, :target_rank].copy()
                W = W_sb[:, :target_rank].copy()
                U += np.random.randn(n*m, target_rank) * 0.05
                V += np.random.randn(m*p, target_rank) * 0.05
                W += np.random.randn(p*n, target_rank) * 0.05
            else:
                # Random init with moderate scale
                U = np.random.randn(n*m, target_rank) * 0.4
                V = np.random.randn(m*p, target_rank) * 0.4
                W = np.random.randn(p*n, target_rank) * 0.4
            
            for it in range(40):
                if time.time() > deadline - 25.0:
                    break
                
                # ALS: Update U
                Vkr = np.zeros((m*p * p*n, target_rank))
                for r in range(target_rank):
                    Vkr[:, r] = np.kron(V[:, r], W[:, r])
                T_flat = T.reshape(n*m, -1)
                VkrT_Vkr = Vkr.T @ Vkr
                U = (T_flat @ Vkr) @ np.linalg.pinv(VkrT_Vkr)
                
                # ALS: Update V
                UW = np.zeros((n*m * p*n, target_rank))
                for r in range(target_rank):
                    UW[:, r] = np.kron(U[:, r], W[:, r])
                UW_T_UW = UW.T @ UW
                V = np.array([(T[:, j, :].reshape(-1) @ UW) @ np.linalg.pinv(UW_T_UW) for j in range(m*p)])
                
                # ALS: Update W
                UV = np.zeros((n*m * m*p, target_rank))
                for r in range(target_rank):
                    UV[:, r] = np.kron(U[:, r], V[:, r])
                UV_T_UV = UV.T @ UV
                W = np.array([(T[:, :, k].reshape(-1) @ UV) @ np.linalg.pinv(UV_T_UV) for k in range(p*n)])
                
                # Check convergence
                res = get_residual(U, V, W)
                if res < 1e-14:
                    if verify(U, V, W, tol=1e-10):
                        best_U = U
                        best_V = V
                        best_W = W
                        best_rank = target_rank
                    break
            
            # Early exit if found at this rank
            if best_rank == target_rank:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
