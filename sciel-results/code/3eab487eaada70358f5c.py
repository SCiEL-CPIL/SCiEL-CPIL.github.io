import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145  # Leave 5s margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    NM, MP, PN = n * m, m * p, p * n  # 12, 24, 18
    
    # Build target tensor T[i,j,k] = 1
    T = np.ones((n, m, p))
    
    def check_exact(U, V, W):
        """Check if U, V, W exactly reconstruct T"""
        # U: (n, m, r), V: (m, p, r), W: (p, n, r)
        # T_rec[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        return np.allclose(T_rec, T, atol=1e-9)

    def als_step_U(U, V, W, T):
        """Update U given V, W"""
        n, m, p, r = U.shape[0], U.shape[1], W.shape[0], U.shape[2]
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # A[k, c] = V[j, k, c] * W[k, i, c]
                A = V[j, :, :] * W[:, i, :]  # shape (p, r)
                b = T[i, j, :]  # shape (p,)
                # Solve A^T A x = A^T b
                try:
                    # Use lstsq for stability
                    U_new[i, j, :], residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    U_new[i, j, :] = U[i, j, :]
        return U_new

    def als_step_V(U, V, W, T):
        """Update V given U, W"""
        n, m, p, r = U.shape[0], U.shape[1], W.shape[0], U.shape[2]
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                # A[i, c] = U[i, j, c] * W[k, i, c]
                A = U[:, j, :] * W[k, :, :]  # shape (n, r)
                b = T[:, j, k]  # shape (n,)
                try:
                    V_new[j, k, :], residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    V_new[j, k, :] = V[j, k, :]
        return V_new

    def als_step_W(U, V, W, T):
        """Update W given U, V"""
        n, m, p, r = U.shape[0], U.shape[1], W.shape[0], U.shape[2]
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                # A[j, c] = U[i, j, c] * V[j, k, c]
                A = U[i, :, :] * V[:, k, :]  # shape (m, r)
                b = T[i, :, k]  # shape (m,)
                try:
                    W_new[k, i, :], residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    W_new[k, i, :] = W[k, i, :]
        return W_new

    def try_rank_decomposition(rank, num_trials=10, max_iters=100):
        """Try to find exact decomposition of given rank using ALS"""
        total_params = (n * m + m * p + p * n) * rank
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(trial * 1000 + 42)
            
            # Initialize with small random values
            U = np.random.randn(n, m, rank) * 0.5
            V = np.random.randn(m, p, rank) * 0.5
            W = np.random.randn(p, n, rank) * 0.5
            
            prev_error = float('inf')
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                
                U = als_step_U(U, V, W, T)
                V = als_step_V(U, V, W, T)
                W = als_step_W(U, V, W, T)
                
                # Calculate error
                T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                error = np.linalg.norm(T_rec - T)
                
                if error < 1e-9:
                    # Found a real solution, now try rounding
                    U_half = np.round(U * 2) / 2
                    V_half = np.round(V * 2) / 2
                    W_half = np.round(W * 2) / 2
                    if check_exact(U_half, V_half, W_half):
                        return (U_half, V_half, W_half)
                    
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    if check_exact(U_int, V_int, W_int):
                        return (U_int, V_int, W_int)
                    
                    # If real solution is exact but not integer/half, try to see if we can
                    # find a nearby integer solution by local search?
                    # For now, return None to try next trial, hoping for a better init
                    # that lands closer to integers.
                    pass
                
                if abs(prev_error - error) < 1e-12:
                    break
                prev_error = error
        
        return None

    # Search for minimal rank
    best_result = None
    
    # Lower bound is max(nm, mp, np) = 24
    # Known upper bound is 72
    # Try ranks from 40 to 71
    # We prioritize lower ranks
    
    for rank in range(40, 72):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_decomposition(rank, num_trials=10, max_iters=100)
        if result is not None:
            U, V, W = result
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            break
    
    # If no rank found in first pass, try a few more with more effort
    if best_result is None:
        for rank in range(35, 50):
            if time.time() > deadline - 10:
                break
            
            result = try_rank_decomposition(rank, num_trials=20, max_iters=200)
            if result is not None:
                U, V, W = result
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
    
    # Fallback to schoolbook
    if best_result is None:
        rank = rank_schoolbook
        U = np.zeros((n, m, rank))
        V = np.zeros((m, p, rank))
        W = np.zeros((p, n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i, j, col] = 1
                    V[j, k, col] = 1
                    W[k, i, col] = 1
                    col += 1
        best_result = {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    return best_result
# EVOLVE-BLOCK-END
