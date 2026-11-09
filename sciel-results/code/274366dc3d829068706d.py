import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    deadline = start_time + 110
    
    def compute_error(U, V, W, r):
        """Compute max error of the trilinear constraint."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def try_rank(r, max_trials=50):
        """Try to find a valid rank-r decomposition."""
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            np.random.seed(42 + trial)
            
            # Initialize with a structured pattern
            # Use a "block diagonal" inspired initialization
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Strategy: Use r terms, each contributing to a subset of (i,j,k)
            # Initialize with random values, then refine
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Gradient descent on the error
            lr = 0.01
            for it in range(500):
                if time.time() > deadline:
                    break
                
                err = compute_error(U, V, W, r)
                if err < 1e-10:
                    break
                
                # Compute gradients numerically
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                # Sample a few (i,j,k) for gradient estimation
                indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
                
                for idx in indices:
                    i, j, k = idx
                    ui = U[i*m+j, :]
                    vj = V[j*p+k, :]
                    wk = W[k*n+i, :]
                    val = np.sum(ui * vj * wk)
                    err_val = val - 1.0
                    
                    # Gradient w.r.t. U[i*m+j, c] = err_val * V[j*p+k, c] * W[k*n+i, c]
                    grad_U[i*m+j, :] += err_val * vj * wk
                    grad_V[j*p+k, :] += err_val * ui * wk
                    grad_W[k*n+i, :] += err_val * ui * vj
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Reduce learning rate
                lr *= 0.995
            
            err = compute_error(U, V, W, r)
            if err < 1e-8:
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if compute_error(U_r, V_r, W_r, r) < 1e-8:
                    return (U_r, V_r, W_r)
                # Also try rounding to integers
                U_r2 = np.round(U)
                V_r2 = np.round(V)
                W_r2 = np.round(W)
                if compute_error(U_r2, V_r2, W_r2, r) < 1e-8:
                    return (U_r2, V_r2, W_r2)
        
        return None
    
    # Try ranks from low to high
    for r in range(20, 41):
        if time.time() > deadline - 5:
            break
        result = try_rank(r)
        if result is not None:
            U_r, V_r, W_r = result
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Schoolbook fallback
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
