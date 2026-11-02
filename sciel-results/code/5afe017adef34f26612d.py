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
    
    # Try a structured approach: split into 2x2 (Strassen rank 7) and 3x3 (Laderman rank 15)
    # plus cross-block corrections
    # For a 5x5 matrix, we can partition into 2+3 blocks
    
    # Actually, let's try a more direct approach using known decompositions
    # The minimum rank for 5x5 matrix multiplication is not known exactly,
    # but we know it's at most 125 (schoolbook) and at least 60 (lower bounds)
    
    # Let's try to find a decomposition with rank around 80-100 using optimization
    
    def try_optimized_decomposition(rank, max_iters=100, tol=1e-6, seed=42):
        """Try to find a rank decomposition via gradient descent."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Learning rate
        lr = 0.01
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Compute gradients
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # dT/dU[i*m+j, c] = V[j*p+k, c] * W[k*n+i, c] for the (i,j,k) where k is summed
            # Actually, for fixed (i,j), T[i,j,k] for all k:
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # dT[i,j,k]/dU[i*m+j, c] = V[j*p+k, c] * W[k*n+i, c]
            
            # Compute error tensor
            E = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        E[i, j, k] = val - 1.0
            
            max_err = np.max(np.abs(E))
            if max_err < tol:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    grad_U = np.zeros(rank)
                    for k in range(p):
                        for c in range(rank):
                            grad_U[c] += E[i, j, k] * V[j*p+k, c] * W[k*n+i, c]
                    U[i*m+j, :] -= lr * grad_U
            
            # Update V
            for j in range(m):
                for k in range(p):
                    grad_V = np.zeros(rank)
                    for i in range(n):
                        for c in range(rank):
                            grad_V[c] += E[i, j, k] * U[i*m+j, c] * W[k*n+i, c]
                    V[j*p+k, :] -= lr * grad_V
            
            # Update W
            for k in range(p):
                for i in range(n):
                    grad_W = np.zeros(rank)
                    for j in range(m):
                        for c in range(rank):
                            grad_W[c] += E[i, j, k] * U[i*m+j, c] * V[j*p+k, c]
                    W[k*n+i, :] -= lr * grad_W
        
        # Verify
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < 1e-6:
            # Try to round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            max_err_r = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c] for c in range(rank))
                        err = abs(val - 1.0)
                        if err > max_err_r:
                            max_err_r = err
            
            if max_err_r < 1e-8:
                return U_r, V_r, W_r, max_err_r
            else:
                return U, V, W, max_err
        return None
    
    # Try various ranks
    for rank in range(90, best_rank, -5):
        if time.time() > deadline - 15:
            break
        
        result = try_optimized_decomposition(rank, max_iters=50, seed=rank)
        if result is not None:
            U, V, W, err = result
            best_rank = rank
            best_result = (U, V, W)
            break
    
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
