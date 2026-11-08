import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition of T."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        T_approx = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        return np.allclose(T_approx, T, atol=1e-8)

    def to_list(arr):
        return [[float(x) for x in row] for row in arr]

    # Fallback: Rank 28 solution from current program
    def get_rank_28_solution():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        # Part 1: A2 B2 (j=2, k=0..4)
        for i in range(n):
            for k in range(p):
                U[i * m + 2][r] = 1
                V[2 * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
        # Part 2: A1 B13 (j=0,1, k=4)
        for i in range(n):
            for j in range(2):
                U[i * m + j][r] = 1
                V[j * p + 4][r] = 1
                W[4 * n + i][r] = 1
                r += 1
        # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen
        # M1
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1
        V[0 * p + 0][r] = 1; V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = 1; W[1 * n + 1][r] = 1
        r += 1
        # M2
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
        V[0 * p + 0][r] = 1
        W[0 * n + 1][r] = 1; W[1 * n + 1][r] = -1
        r += 1
        # M3
        U[0 * m + 0][r] = 1
        V[0 * p + 1][r] = 1; V[1 * p + 1][r] = -1
        W[1 * n + 0][r] = 1; W[1 * n + 1][r] = 1
        r += 1
        # M4
        U[1 * m + 1][r] = 1
        V[1 * p + 0][r] = 1; V[0 * p + 0][r] = -1
        W[0 * n + 0][r] = 1; W[0 * n + 1][r] = 1
        r += 1
        # M5
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
        V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = -1; W[1 * n + 0][r] = 1
        r += 1
        # M6
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1
        V[0 * p + 0][r] = 1; V[0 * p + 1][r] = 1
        W[1 * n + 1][r] = 1
        r += 1
        # M7
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1
        V[1 * p + 0][r] = 1; V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = 1
        r += 1
        # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen
        # M1
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1
        V[0 * p + 2][r] = 1; V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = 1; W[3 * n + 1][r] = 1
        r += 1
        # M2
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
        V[0 * p + 2][r] = 1
        W[2 * n + 1][r] = 1; W[3 * n + 1][r] = -1
        r += 1
        # M3
        U[0 * m + 0][r] = 1
        V[0 * p + 3][r] = 1; V[1 * p + 3][r] = -1
        W[3 * n + 0][r] = 1; W[3 * n + 1][r] = 1
        r += 1
        # M4
        U[1 * m + 1][r] = 1
        V[1 * p + 2][r] = 1; V[0 * p + 2][r] = -1
        W[2 * n + 0][r] = 1; W[2 * n + 1][r] = 1
        r += 1
        # M5
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
        V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = -1; W[3 * n + 0][r] = 1
        r += 1
        # M6
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1
        V[0 * p + 2][r] = 1; V[0 * p + 3][r] = 1
        W[3 * n + 1][r] = 1
        r += 1
        # M7
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1
        V[1 * p + 2][r] = 1; V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = 1
        r += 1
        
        return {"U": U, "V": V, "W": W}
    
    def try_als_for_rank(target_rank, max_iters=200, n_restarts=10):
        if time.time() > deadline:
            return None
        
        best_solution = None
        
        # Precompute flattened T for faster access
        T_flat_U = T.reshape(n * m, -1)
        T_flat_V = T.reshape(m * p, -1)
        T_flat_W = T.reshape(p * n, -1)
        
        # Dimensions of the "basis" matrices
        dim_B_U = m * p * p * n
        dim_B_V = n * m * p * n
        dim_B_W = n * m * m * p
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(1000 + restart)
            # Initialize with random noise
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Normalize to prevent overflow/underflow
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U
                # Construct B_mat_T for U update: shape (dim_B_U, target_rank)
                # B_mat_T[:, r] = (V[:, r] * W[:, r]).flatten()
                # We can do this efficiently using broadcasting
                # V[:, r] is (m*p, 1), W[:, r] is (p*n, 1)
                # Outer product is (m*p, p*n)
                # We need to stack these for all r.
                # V is (m*p, R), W is (p*n, R)
                # We want a tensor of shape (m*p, p*n, R) where T[a, b, r] = V[a, r] * W[b, r]
                # Then reshape to (m*p*p*n, R)
                
                # Efficient construction:
                # V[:, :, None] * W[None, :, :] -> (m*p, p*n, R)
                # Reshape to (m*p*p*n, R)
                B_mat_T_U = (V[:, :, None] * W[None, :, :]).reshape(-1, target_rank)
                
                # Solve U = T_flat_U @ pinv(B_mat_T_U)
                # Using lstsq is better
                try:
                    U_new, _, _, _ = np.linalg.lstsq(B_mat_T_U, T_flat_U.T, rcond=None)
                    U = U_new.T
                except:
                    break
                
                # Update V
                B_mat_T_V = (U[:, :, None] * W[None, :, :]).reshape(-1, target_rank)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(B_mat_T_V, T_flat_V.T, rcond=None)
                    V = V_new.T
                except:
                    break
                
                # Update W
                B_mat_T_W = (U[:, :, None] * V[None, :, :]).reshape(-1, target_rank)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(B_mat_T_W, T_flat_W.T, rcond=None)
                    W = W_new.T
                except:
                    break
                
                # Check convergence
                if verify(U, V, W):
                    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
                
                # Optional: Check error
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.max(np.abs(T_approx - T))
                if err < 1e-6:
                    # Polish
                    for _ in range(5):
                        if time.time() > deadline:
                            break
                        B_mat_T_U = (V[:, :, None] * W[None, :, :]).reshape(-1, target_rank)
                        U_new, _, _, _ = np.linalg.lstsq(B_mat_T_U, T_flat_U.T, rcond=None)
                        U = U_new.T
                        
                        B_mat_T_V = (U[:, :, None] * W[None, :, :]).reshape(-1, target_rank)
                        V_new, _, _, _ = np.linalg.lstsq(B_mat_T_V, T_flat_V.T, rcond=None)
                        V = V_new.T
                        
                        B_mat_T_W = (U[:, :, None] * V[None, :, :]).reshape(-1, target_rank)
                        W_new, _, _, _ = np.linalg.lstsq(B_mat_T_W, T_flat_W.T, rcond=None)
                        W = W_new.T
                        
                        if verify(U, V, W):
                            return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
                            
        return None

    # Try to find a rank 27 solution
    sol_27 = try_als_for_rank(27, max_iters=100, n_restarts=5)
    if sol_27:
        return sol_27
    
    # Try rank 26
    sol_26 = try_als_for_rank(26, max_iters=50, n_restarts=3)
    if sol_26:
        return sol_26
        
    # If not, return the known rank 28 solution
    return get_rank_28_solution()

# EVOLVE-BLOCK-END
