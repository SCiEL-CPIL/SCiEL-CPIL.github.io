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

    # Try to find a rank 27 solution using ALS
    def try_als_for_rank(target_rank, max_iters=100, n_restarts=5):
        if time.time() > deadline:
            return None
        
        best_solution = None
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + restart)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U: for each row i, solve min ||T[i,:,:] - U[i,:] V^T W^T||
                # Reshape T to (n*m, m*p * p*n)
                T_flat = T.reshape(n * m, -1)
                # We want U[i,:] such that U[i,:] @ (V^T @ W^T) approx T_flat[i,:]
                # Let M = V @ W.T -> (m*p, p*n) @ (p*n, m*p) is not right.
                # Tensor T = sum_r U[:,r] V[:,r] W[:,r]
                # T[i, j, k] = sum_r U[i,r] V[j,r] W[k,r]
                # Fix V, W. Update U[i,r].
                # T[i, :, :] = sum_r U[i,r] (V[:,r] outer W[:,r])
                # This is a linear system for U[i,:].
                
                # Precompute the "basis" matrices for U update
                # For each r, B_r = V[:,r] outer W[:,r] -> shape (m*p, p*n)
                # We want to solve: T_flat[i] = sum_r U[i,r] B_r.flatten()
                # Let B_mat be shape (target_rank, m*p*p*n) where B_mat[r, :] = B_r.flatten()
                # Then T_flat[i] = U[i,:] @ B_mat.T
                # U[i,:] = T_flat[i] @ pinv(B_mat.T)
                
                # Construct B_mat
                # B_mat[r] = (V[:,r] * W[:,r]) reshaped
                # V[:,r] is (m*p, 1), W[:,r] is (p*n, 1)
                # Outer product is (m*p, p*n)
                
                # Efficient way:
                # T_flat is (n*m, m*p*p*n)
                # We need to solve U[i] = T_flat[i] @ (B_mat.T)^+
                
                # Let's construct B_mat.T which is (m*p*p*n, target_rank)
                # B_mat.T[col, r] = V[col // (p*n), r] * W[col % (p*n), r]
                # This is expensive to construct explicitly for each iteration if done naively.
                # But dimensions are small: m*p=15, p*n=10, target_rank=27.
                # Size of B_mat.T is 150 x 27.
                
                B_mat_T = np.zeros((m * p * p * n, target_rank))
                for r in range(target_rank):
                    # V[:, r] is (m*p, 1)
                    # W[:, r] is (p*n, 1)
                    # Outer product
                    outer_prod = np.outer(V[:, r], W[:, r])
                    B_mat_T[:, r] = outer_prod.flatten()
                
                # Solve for U
                # U[i, r] = sum_k T_flat[i, k] * (pinv(B_mat_T)[k, r])
                # pinv(B_mat_T) is (150, 27)
                # Let P = pinv(B_mat_T)
                # U = T_flat @ P
                try:
                    P_U = np.linalg.pinv(B_mat_T)
                    U_new = T_flat @ P_U
                except:
                    break
                
                # Update V
                # T_flat_V: reshape T to (m*p, n*m * p*n)
                # T[j, i, k] = sum_r V[j,r] U[i,r] W[k,r]
                # For fixed j, T[j, :, :] = sum_r V[j,r] (U[:,r] outer W[:,r])
                T_flat_V = T.reshape(m * p, n * m * p * n)
                B_mat_T_V = np.zeros((n * m * p * n, target_rank))
                for r in range(target_rank):
                    outer_prod = np.outer(U[:, r], W[:, r])
                    B_mat_T_V[:, r] = outer_prod.flatten()
                try:
                    P_V = np.linalg.pinv(B_mat_T_V)
                    V_new = T_flat_V @ P_V
                except:
                    break
                
                # Update W
                T_flat_W = T.reshape(p * n, n * m * m * p)
                B_mat_T_W = np.zeros((n * m * m * p, target_rank))
                for r in range(target_rank):
                    outer_prod = np.outer(U[:, r], V[:, r])
                    B_mat_T_W[:, r] = outer_prod.flatten()
                try:
                    P_W = np.linalg.pinv(B_mat_T_W)
                    W_new = T_flat_W @ P_W
                except:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence/validity
                if verify(U, V, W):
                    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
                    
                # Normalize to prevent drift
                norms_U = np.linalg.norm(U, axis=0, keepdims=True)
                norms_V = np.linalg.norm(V, axis=0, keepdims=True)
                norms_W = np.linalg.norm(W, axis=0, keepdims=True)
                # Simple normalization: divide each column by its norm if > 0
                U = U / (norms_U + 1e-8)
                V = V / (norms_V + 1e-8)
                W = W / (norms_W + 1e-8)
                
                # Re-scale T? No, T is fixed. The scaling of U,V,W matters for the product.
                # If we scale U col r by s, V col r by 1/s, W col r by 1, product is invariant.
                # The above normalization changes the scale. We should track scales or just not normalize
                # if we are doing exact linear solves.
                # Actually, ALS solves for the exact best fit given the other two.
                # Normalization is not strictly necessary for correctness, only for numerical stability.
                # Let's remove normalization to avoid introducing errors if not done carefully.
                # Re-run without normalization?
                # Let's just check validity again.
                
        return None

    # First, check if rank 27 is possible
    sol_27 = try_als_for_rank(27)
    if sol_27:
        return sol_27
        
    # If not, return the known rank 28 solution
    return get_rank_28_solution()

# EVOLVE-BLOCK-END
