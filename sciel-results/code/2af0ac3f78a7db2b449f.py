import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Build the schoolbook decomposition as a safe fallback
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = (U_school, V_school, W_school)
    
    def try_optimized_decomposition(rank, max_iters=200, tol=1e-8, seed=42):
        """Try to find a rank decomposition via gradient descent (vectorized)."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        
        # Initialize with small random values
        # Using (n, m, rank) etc for easier einsum
        U = np.random.randn(n, m, rank) * 0.1
        V = np.random.randn(m, p, rank) * 0.1
        W = np.random.randn(p, n, rank) * 0.1
        
        # Learning rate
        lr = 0.05
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Compute T tensor
            # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
            T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            
            # Compute error
            E = T - 1.0
            
            max_err = np.max(np.abs(E))
            if max_err < tol:
                break
            
            # Compute gradients
            # dE/dU[i,j,c] = sum_k V[j,k,c] * W[k,i,c] * E[i,j,k]
            grad_U = np.einsum('jkc,kic,ijk->ijc', V, W, E)
            
            # dE/dV[j,k,c] = sum_i U[i,j,c] * W[k,i,c] * E[i,j,k]
            grad_V = np.einsum('ijc,kic,ijk->jkc', U, W, E)
            
            # dE/dW[k,i,c] = sum_j U[i,j,c] * V[j,k,c] * E[i,j,k]
            grad_W = np.einsum('ijc,jkc,ijk->kic', U, V, E)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Optional: decay learning rate
            # lr *= 0.999
        
        # Verify
        T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        max_err = np.max(np.abs(T - 1.0))
        
        if max_err < 1e-6:
            # Try to round to half-integers
            # Flatten back to (n*m, rank) etc for output
            U_flat = U.reshape(n*m, rank)
            V_flat = V.reshape(m*p, rank)
            W_flat = W.reshape(p*n, rank)
            
            U_r = np.round(U_flat * 2) / 2
            V_r = np.round(V_flat * 2) / 2
            W_r = np.round(W_flat * 2) / 2
            
            # Verify rounded
            T_r = np.einsum('ijc,jkc,kic->ijk', U_r.reshape(n,m,rank), V_r.reshape(m,p,rank), W_r.reshape(p,n,rank))
            max_err_r = np.max(np.abs(T_r - 1.0))
            
            if max_err_r < 1e-8:
                return U_r, V_r, W_r, max_err_r
            else:
                # If rounding fails, check if raw is good enough? 
                # The task requires integer/half-integer entries.
                # So if rounding fails, it's invalid.
                return None
        return None
    
    # Try various ranks, step by 1 to catch 124, 123, etc.
    # We have time for many attempts now due to vectorization.
    for rank in range(rank_schoolbook - 1, 89, -1):
        if time.time() > deadline - 15:
            break
        
        # Try a few seeds for each rank
        found = False
        for seed in [rank, rank + 1, rank + 100]:
            if time.time() > deadline - 5:
                break
            result = try_optimized_decomposition(rank, max_iters=100, seed=seed)
            if result is not None:
                U, V, W, err = result
                best_rank = rank
                best_result = (U, V, W)
                found = True
                break
        
        if found:
            # If we found a lower rank, we can stop or continue to find even lower?
            # The task asks for rank-minimal. So we should continue to lower ranks.
            # But if we found 124, we should try 123.
            # The loop naturally continues to rank-1.
            pass
    
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
