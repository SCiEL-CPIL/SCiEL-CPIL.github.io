import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a tensor decomposition for matrix multiplication of dimensions 4x5 * 5x5.
    
    The schoolbook rank is n*m*p = 100. We attempt to find a lower-rank decomposition
    using gradient-based optimization. If no improvement is found within the time budget,
    we return the schoolbook decomposition.
    
    The tensor equation is:
    sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'}
    
    for all i in [0,n), j,j' in [0,m), k in [0,p).
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard: 120s wall-clock limit, leave 10s margin
    deadline = time.time() + 110
    
    def compute_error(U, V, W, R):
        """Compute the squared error of the decomposition."""
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # For each (i, j, jp, k), compute sum_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r]
        # Target: 1.0 if j == jp, else 0.0
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    target = 1.0 if j == jp else 0.0
                    for k in range(p):
                        s = 0.0
                        for r in range(R):
                            s += U[i*m + j, r] * V[jp*p + k, r] * W[k*n + i, r]
                        error += (s - target) ** 2
        return error
    
    def compute_gradients(U, V, W, R):
        """Compute gradients of the squared error with respect to U, V, W."""
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    target = 1.0 if j == jp else 0.0
                    for k in range(p):
                        # Compute the current sum
                        s = 0.0
                        for r in range(R):
                            s += U[i*m + j, r] * V[jp*p + k, r] * W[k*n + i, r]
                        diff = 2.0 * (s - target)
                        
                        for r in range(R):
                            grad_U[i*m + j, r] += diff * V[jp*p + k, r] * W[k*n + i, r]
                            grad_V[jp*p + k, r] += diff * U[i*m + j, r] * W[k*n + i, r]
                            grad_W[k*n + i, r] += diff * U[i*m + j, r] * V[jp*p + k, r]
        
        return grad_U, grad_V, grad_W
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Try ranks from schoolbook-1 down to some minimum
    # For 4x5*5x5, the known lower bound is 98 (Winograd-style improvements)
    # We'll try ranks 99, 98, 97, etc.
    
    for R in range(rank_schoolbook - 1, max(rank_schoolbook - 5, 0), -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random restarts for each rank
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + R * 100 + restart)
            
            # Initialize with small random values
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            # Gradient descent with adaptive learning rate
            lr = 0.01
            best_error = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(2000):
                if time.time() > deadline - 10:
                    break
                
                # Compute error
                err = compute_error(U, V, W, R)
                
                if err < best_error:
                    best_error = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-12:
                    break
                
                # Compute gradients
                grad_U, grad_V, grad_W = compute_gradients(U, V, W, R)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Adaptive learning rate: reduce if error increases
                if err > best_error * 1.1:
                    lr *= 0.5
                
                # Increase learning rate slowly if error is decreasing
                if err < best_error * 0.9 and lr < 0.05:
                    lr *= 1.05
            
            # Check if we found a good solution
            if best_error < 1e-10:
                # Round to half-integers
                U_rounded = np.round(best_U * 2) / 2
                V_rounded = np.round(best_V * 2) / 2
                W_rounded = np.round(best_W * 2) / 2
                
                # Verify the rounded solution
                err_rounded = compute_error(U_rounded, V_rounded, W_rounded, R)
                if err_rounded < 1e-6:
                    # Found a valid lower-rank decomposition
                    if R < best_rank:
                        best_rank = R
                        best_result = {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                    break
        else:
            continue
        break
    
    # If no lower-rank decomposition found, return schoolbook
    if best_result is not None:
        return best_result
    
    # Schoolbook decomposition
    U = [[0.0] * rank_schoolbook for _ in range(n * m)]
    V = [[0.0] * rank_schoolbook for _ in range(m * p)]
    W = [[0.0] * rank_schoolbook for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1.0
                V[j * p + k][idx] = 1.0
                W[k * n + i][idx] = 1.0
                idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
