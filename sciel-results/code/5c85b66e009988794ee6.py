import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _als_decompose(n, m, p, target_rank, max_iter=500, tol=1e-9, time_limit=100.0, seed=42):
    """
    Attempts to find a rank-`target_rank` decomposition using Alternating Least Squares.
    Returns (U, V, W) if successful, else None.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Dimensions
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    R = target_rank
    
    # Target tensor vector t of length n*m*p, all ones
    # T[i, j, k] = 1
    # We reshape to match the flattening order of the contraction.
    # The contraction is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We can view this as:
    # For each r, we have a rank-1 tensor T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want sum_r T_r = T_target (all ones)
    
    # ALS updates:
    # Given V, W, find U.
    # U[:, r] is a vector.
    # The equation for a fixed r is not linear in U because V and W are coupled in the sum.
    # However, if we fix V and W, the contribution of U[:, r] to the total sum is linear?
    # No, the total sum is sum_r U_r * V_r * W_r.
    # If we fix V and W, and we want to update U[:, r] for a specific r, we can't isolate it easily unless we treat the other terms as a residual.
    
    # Standard ALS for CP decomposition:
    # Minimize || T - sum_r u_r v_r^T w_r^T ||_F
    # Update u_r: u_r = (T x_2 W_r x_1 V_r^T) / ( ||w_r||^2 ||v_r||^2 ) ... this is for general T.
    
    # Here T is the all-ones tensor.
    # Let's use the property that T_target is all ones.
    
    # Residual approach:
    # Let S = sum_{r' != r} U[:, r'] * V[:, r'] * W[:, r']
    # We want U[:, r] * V[:, r] * W[:, r] = T_target - S
    # This is still non-linear for U, V, W simultaneously.
    
    # Better ALS:
    # Fix V, W.
    # We want to find U such that sum_r U[:, r] * (V[:, r] * W[:, r])_reshaped = T_target.
    # Let M_r be the matrix formed by V[:, r] and W[:, r] such that (U[:, r] * M_r) = T_target contribution.
    # Actually, let's flatten the tensor equation.
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # For a fixed r, let's define the interaction.
    # This is tricky because the indices are permuted.
    
    # Alternative: Use gradient descent on the parameters U, V, W.
    # Loss = sum_{i,j,k} (1 - sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r])^2
    
    # We will implement a simple gradient descent / Adam optimizer.
    
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(dim_u, R) * 0.1
    V = rng.randn(dim_v, R) * 0.1
    W = rng.randn(dim_w, R) * 0.1
    
    # Precompute indices for fast evaluation
    # We need to compute the tensor T_est
    # T_est[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # We can vectorize this using einsum or matmul tricks.
    # Let's use a loop over r for clarity first, then optimize if needed.
    # But 100s is a lot of time. We can do many iterations.
    
    # Let's try to be smarter.
    # For each r, the term is an outer product of vectors in different spaces, permuted.
    
    # Let's use a simple SGD with momentum.
    lr = 0.01
    momentum = 0.9
    v_u = np.zeros_like(U)
    v_v = np.zeros_like(V)
    v_w = np.zeros_like(W)
    
    best_loss = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for it in range(max_iter):
        if time.time() > deadline - 10:
            break
            
        # Compute Loss and Gradients
        T_est = np.zeros((n, m, p))
        
        # Gradient for U:
        # dL/dU[i*m+j, r] = 2 * (T_est[i,j,k] - 1) * V[j*p+k, r] * W[k*n+i, r] summed over k?
        # No, T_est[i,j,k] depends on U[i*m+j, r].
        # L = sum_{i,j,k} (1 - T_est[i,j,k])^2
        # dL/dU[i*m+j, r] = sum_k 2 * (T_est[i,j,k] - 1) * (- V[j*p+k, r] * W[k*n+i, r])
        
        # This is O(n*m*p*R) per iteration.
        # n=2, m=3, p=3. n*m*p = 18. R=17.
        # 18 * 17 = 306 operations per gradient element?
        # Total elements in U: 6 * 17 = 102.
        # Total elements in V: 9 * 17 = 153.
        # Total elements in W: 6 * 17 = 102.
        # Total ~ 357 elements.
        # Gradient calculation:
        # For each U element (i,j,r), we sum over k (3 terms).
        # So 102 * 3 = 306 mults for U grad.
        # Similarly for V and W.
        # Total ~ 1000 ops per iteration.
        # 500 iterations is 500,000 ops. Very fast.
        # We can increase max_iter significantly.
        
        # Let's implement the gradient computation.
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Compute T_est first
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    T_est[i, j, k] = val
        
        loss = np.sum((T_est - 1.0)**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
        if loss < tol:
            return best_U, best_V, best_W
            
        # Compute Gradients
        # dL/dU[i*m+j, r] = -2 * sum_k (T_est[i,j,k] - 1) * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for r in range(R):
                    g = 0.0
                    for k in range(p):
                        g += (T_est[i, j, k] - 1.0) * V[j*p+k, r] * W[k*n+i, r]
                    grad_U[i*m+j, r] = -2.0 * g
                    
        # dL/dV[j*p+k, r] = -2 * sum_i (T_est[i,j,k] - 1) * U[i*m+j, r] * W[k*n+i, r]
        for j in range(m):
            for k in range(p):
                for r in range(R):
                    g = 0.0
                    for i in range(n):
                        g += (T_est[i, j, k] - 1.0) * U[i*m+j, r] * W[k*n+i, r]
                    grad_V[j*p+k, r] = -2.0 * g
                    
        # dL/dW[k*n+i, r] = -2 * sum_j (T_est[i,j,k] - 1) * U[i*m+j, r] * V[j*p+k, r]
        for k in range(p):
            for i in range(n):
                for r in range(R):
                    g = 0.0
                    for j in range(m):
                        g += (T_est[i, j, k] - 1.0) * U[i*m+j, r] * V[j*p+k, r]
                    grad_W[k*n+i, r] = -2.0 * g
        
        # Update with Momentum
        v_u = momentum * v_u + (1 - momentum) * grad_U
        v_v = momentum * v_v + (1 - momentum) * grad_V
        v_w = momentum * v_w + (1 - momentum) * grad_W
        
        U -= lr * v_u
        V -= lr * v_v
        W -= lr * v_w
        
        # Optional: Normalize to prevent divergence
        # for r in range(R):
        #     norm = (np.linalg.norm(U[:,r]) * np.linalg.norm(V[:,r]) * np.linalg.norm(W[:,r])) ** (1/3)
        #     if norm > 1e-6:
        #         U[:,r] /= norm
        #         V[:,r] /= norm
        #         W[:,r] /= norm
        #         # Scale back? No, this changes the scale. 
        #         # Better to just let it be.
        
    return None

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(R):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Schoolbook rank is 18
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Try to find lower rank solutions using Gradient Descent
    # We try ranks 17, 16, 15
    # We use multiple random restarts for each rank
    
    for target_rank in [17, 16, 15]:
        if time.time() > deadline - 30:
            break
            
        found = False
        # Try a few random seeds/restarts
        for seed in range(10):
            if time.time() > deadline - 20:
                break
                
            result = _als_decompose(n, m, p, target_rank, max_iter=2000, tol=1e-9, time_limit=deadline - time.time() - 15, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
