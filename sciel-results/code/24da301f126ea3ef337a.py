"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    
    The multiplication tensor T has indices (i, j, k) where:
    - i in [0, n) = [0, 3): row of A, column of W (first index)
    - j in [0, m) = [0, 3): column of A, row of B
    - k in [0, p) = [0, 6): column of B, second index of W
    
    T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    
    We seek rank r such that:
    T = sum_{c=0}^{r-1} U[:,c] outer V[:,c] outer W[:,c]
    
    where U is (n*m, r), V is (m*p, r), W is (p*n, r)
    
    The schoolbook decomposition has rank n*m*p = 3*3*6 = 54.
    We try to find a lower rank decomposition.
    
    For matrix multiplication M_{n,m,p}, the minimum rank is known to be:
    - For square n=m=p: at least n^2, and the best known is around n^2 * (some factor)
    - For rectangular: at least max(n*m, m*p, p*n)
    
    For 3x3x6: lower bound is max(9, 18, 18) = 18.
    The schoolbook gives 54.
    
    Let's try a systematic search for low-rank decompositions using numerical methods.
    We'll use the approach of finding a low-rank approximation to the tensor.
    """
    n, m, p = DIMENSIONS
    
    # Build the tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try to find a low-rank decomposition
    # Start with the schoolbook rank and try to reduce
    
    # First, let's try a simple approach: use the known structure
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1
    
    # This can be seen as: for each (i,j,k), we have a rank-1 term
    # u_{i,j} * v_{j,k} * w_{k,i}
    
    # A key insight: if we can group terms, we might reduce the rank.
    # For example, if we fix i and vary j,k, we get a matrix of size m x p
    # that is all ones. This matrix has rank 1.
    
    # So for each i, the "slice" T[i*m+j, j*p+k, k*n+i] as a function of (j,k)
    # is the all-ones matrix of size m x p, which has rank 1.
    
    # This suggests we might be able to use rank n * 1 * something...
    # But we need to be careful about the interactions between different i values.
    
    # Let's try a different approach: use alternating least squares or similar
    # to find a low-rank decomposition.
    
    deadline = time.time() + 120  # 120 second deadline with safety margin
    
    # Try ranks from the lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # But we know from theory that for 3x3x3, the minimum rank is at least 19
    # (actually, it's known to be between 19 and 23 for 3x3x3)
    
    # For 3x3x6, let's start with a reasonable guess
    # The schoolbook is 54. Let's try to find something lower.
    
    # Strategy: Use a randomized initialization and alternating minimization
    # to find a low-rank decomposition.
    
    def try_rank(r, max_iters=1000, tol=1e-10):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(42)  # Fixed seed for reproducibility
        
        # Initialize with random matrices
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Scale to have reasonable magnitudes
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each column c, U[:,c] should minimize
            # sum_{i,j,k} (T[i*m+j, j*p+k, k*n+i] - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
            # holding V and W fixed.
            # This is a linear least squares problem.
            
            # Build the design matrix for U
            # For each (i,j), we need to solve for U[i*m+j, :]
            # The equation is: T[i*m+j, :, :] = U[i*m+j, :] @ (V[:, :] * W[:, :].T).T
            # Wait, let me think more carefully.
            
            # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
            
            # For fixed a = i*m+j, we want to find U[a, :] such that:
            # T[a, :, :] ≈ U[a, :] @ (V * W.T)  where * is element-wise
            # Let M = V * W.T (element-wise product), shape (m*p, p*n)
            # Then T[a, :, :] ≈ U[a, :] @ M
            # This is a linear system: U[a, :] = T[a, :, :] @ M.T @ (M @ M.T)^{-1}
            
            M = V * W.T  # shape (m*p, p*n)
            MM = M @ M.T  # shape (m*p, m*p)
            
            # Regularize for numerical stability
            reg = 1e-10 * np.eye(m * p)
            MM_reg = MM + reg
            
            try:
                # Solve for U
                # U[a, :] = T[a, :, :] @ M.T @ inv(MM_reg)
                # This is equivalent to: U = T @ M.T @ inv(MM_reg)
                # But T is 3D, so we need to do this for each a.
                
                # Reshape T to (n*m, m*p, p*n)
                # For each a, solve: U[a, :] @ M = T[a, :, :]
                # U[a, :] = T[a, :, :] @ M.T @ inv(MM_reg)
                
                # Vectorized: U = einsum('abc,cb,dc->ad', T, M, inv(MM_reg))
                # Actually, let's do it more carefully.
                
                # T[a, b, c] = sum_r U[a, r] * M[b, c] * delta... no wait.
                # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
                # Let N[b, r] = V[b, r] * W[c, r]... no, that doesn't work because c varies.
                
                # Better: for fixed a, the equation is:
                # T[a, :, :] = U[a, :] @ (V * W.T)
                # where (V * W.T)[b, c] = V[b, r] * W[c, r] summed over r? No.
                
                # Actually, U[a, :] @ (V * W.T) doesn't make sense dimensionally.
                # Let me redefine.
                
                # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
                
                # For fixed a, define S[b, c] = T[a, b, c]
                # We want: S[b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
                # Let P = V * W.T (element-wise), so P[b, r] = V[b, r], and we need W[c, r].
                # This is not a simple matrix product.
                
                # Alternative: reshape.
                # S = U[a, :] outer (V.T * W) ... no.
                
                # Let's use: S[b, c] = sum_r (U[a, r] * V[b, r]) * W[c, r]
                # Let Q[b, r] = U[a, r] * V[b, r]
                # Then S = Q @ W.T
                # So Q = S @ W.T @ inv(W @ W.T)
                # And U[a, r] = Q[b, r] / V[b, r] for each b... but this must hold for all b.
                
                # This is getting complicated. Let's use a different approach.
                
                # Use the fact that this is a linear least squares problem in U[a, :].
                # The design matrix for U[a, :] is:
                # For each (b, c), the coefficient of U[a, r] is V[b, r] * W[c, r]
                # So the design matrix D has shape (m*p*p*n, r) where
                # D[(b, c), r] = V[b, r] * W[c, r]
                # And the target is T[a, :, :] reshaped to (m*p*p*n,)
                
                # This is a large matrix, but for our sizes it should be manageable.
                # m*p = 18, p*n = 18, so D has shape (18*18, r) = (324, r)
                
                D = (V.T * W).T  # shape (m*p, p*n, r) -> need to reshape
                
                # Actually, let's build D properly.
                # D[(b, c), r] = V[b, r] * W[c, r]
                # We can compute this as: D = (V[:, :, None] * W[None, :, :]).reshape(m*p, p*n, r)
                # Then reshape to (m*p*p*n, r)
                
                D3 = V[:, :, None] * W[None, :, :]  # shape (m*p, p*n, r)
                D_flat = D3.reshape(m * p * p * n, r)
                
                target = T[a].reshape(-1)  # shape (m*p*p*n,)
                
                # Solve D_flat @ U[a, :] = target
                # U[a, :] = lstsq(D_flat, target)
                
                U[a, :], residuals, rank, sv = np.linalg.lstsq(D_flat, target, rcond=None)
                
            except Exception:
                break
            
            # Similarly update V and W
            # For V: T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
            # For fixed b, define S'[a, c] = T[:, b, :]
            # S'[a, c] = sum_r (U[a, r] * W[c, r]) * V[b, r]
            # Let Q'[a, r] = U[a, r] * W[c, r]... no, c varies.
            
            # Similar approach:
            # D_V[(a, c), r] = U[a, r] * W[c, r]
            D_V3 = U[:, :, None] * W[None, :, :]  # shape (n*m, p*n, r)
            D_V_flat = D_V3.reshape(n * m * p * n, r)
            
            for b in range(m * p):
                if time.time() > deadline - 10:
                    break
                target_V = T[:, b, :].reshape(-1)
                V[b, :], residuals, rank, sv = np.linalg.lstsq(D_V_flat, target_V, rcond=None)
            
            # For W: T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
            # For fixed c, define S''[a, b] = T[:, :, c]
            # S''[a, b] = sum_r (U[a, r] * V[b, r]) * W[c, r]
            D_W3 = U[:, :, None] * V[:, None, :]  # shape (n*m, m*p, r)
            D_W_flat = D_W3.reshape(n * m * m * p, r)
            
            for c in range(p * n):
                if time.time() > deadline - 10:
                    break
                target_W = T[:, :, c].reshape(-1)
                W[c, :], residuals, rank, sv = np.linalg.lstsq(D_W_flat, target_W, rcond=None)
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        pred = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                        error += (1.0 - pred) ** 2
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < tol:
                break
        
        if best_error < 1e-8:
            return best_U, best_V, best_W
        return None
    
    # Try ranks from 18 up to 54
    for r in range(18, 55):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = [[float(U[i, j]) for j in range(r)] for i in range(n * m)]
            V_list = [[float(V[i, j]) for j in range(r)] for i in range(m * p)]
            W_list = [[float(W[i, j]) for j in range(r)] for i in range(p * n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # If we didn't find a low-rank decomposition, return the schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
