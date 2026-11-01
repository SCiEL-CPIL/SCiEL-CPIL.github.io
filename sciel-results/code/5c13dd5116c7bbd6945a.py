"""Optimized matrix multiplication tensor decomposition for 2x3 by 3x6."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing factor matrices
    that exactly decompose the matrix multiplication tensor.
    
    Strategy: Start with schoolbook (rank 36), then use gradient-based optimization
    to find lower-rank decompositions with half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    target_rank = n * m * p  # 36 for schoolbook
    
    # Build schoolbook solution as fallback
    U_sb = np.zeros((n * m, target_rank), dtype=float)
    V_sb = np.zeros((m * p, target_rank), dtype=float)
    W_sb = np.zeros((p * n, target_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = {
        'U': U_sb.tolist(),
        'V': V_sb.tolist(),
        'W': W_sb.tolist()
    }
    best_rank = target_rank
    
    # The target tensor T[i,j,k] = 1 for all valid (i,j,k)
    T = np.ones((n, m, p), dtype=float)
    
    # Helper: verify a decomposition
    def verify(U, V, W, r):
        """Check if U, V, W of rank r exactly decompose the tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    # Helper: compute objective (sum of squared errors)
    def objective(U, V, W, r):
        """Compute sum of squared errors for the decomposition."""
        obj = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    obj += (s - 1.0) ** 2
        return obj
    
    # Helper: gradient descent step
    def gradient_step(U, V, W, r, lr):
        """Perform one gradient descent step."""
        # Compute residuals
        residuals = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    residuals[i, j, k] = s - 1.0
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for c in range(r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        grad = 2.0 * residuals[i, j, k] * V[j * p + k, c] * W[k * n + i, c]
                        dU[i * m + j, c] += grad
                for k in range(p):
                    for i in range(n):
                        grad = 2.0 * residuals[i, j, k] * U[i * m + j, c] * W[k * n + i, c]
                        dV[j * p + k, c] += grad
                for i in range(n):
                    grad = 2.0 * residuals[i, j, k] * U[i * m + j, c] * V[j * p + k, c]
                    dW[k * n + i, c] += grad
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        return U, V, W
    
    # Try to find lower rank decompositions
    # Lower bound: at least max(n*m, m*p, p*n) = 18, but realistically higher
    # Try ranks from 20 up to 35
    for r in range(20, target_rank):
        if time.time() > deadline - 15:
            break
        
        # Multiple random initializations
        max_trials = 30 if r >= 28 else 15
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small values
            np.random.seed(12345 + trial * 7 + r)
            scale = 0.5
            U_rand = np.random.randn(n * m, r) * scale
            V_rand = np.random.randn(m * p, r) * scale
            W_rand = np.random.randn(p * n, r) * scale
            
            # Gradient descent
            lr = 0.01
            min_lr = 1e-6
            max_iters = 2000
            
            prev_obj = objective(U_rand, V_rand, W_rand, r)
            
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    break
                
                U_rand, V_rand, W_rand = gradient_step(U_rand, V_rand, W_rand, r, lr)
                
                obj = objective(U_rand, V_rand, W_rand, r)
                
                if obj < 1e-8:
                    # Found a solution! Round to nearest half-integer
                    U_rounded = np.round(U_rand * 2) / 2
                    V_rounded = np.round(V_rand * 2) / 2
                    W_rounded = np.round(W_rand * 2) / 2
                    
                    # Verify the rounded solution
                    if verify(U_rounded, V_rounded, W_rounded, r):
                        best_rank = r
                        best_result = {
                            'U': U_rounded.tolist(),
                            'V': V_rounded.tolist(),
                            'W': W_rounded.tolist()
                        }
                        break
                    else:
                        # Try without rounding
                        if verify(U_rand, V_rand, W_rand, r):
                            best_rank = r
                            best_result = {
                                'U': U_rand.tolist(),
                                'V': V_rand.tolist(),
                                'W': W_rand.tolist()
                            }
                            break
                    break
                
                # Adaptive learning rate
                if obj > prev_obj * 1.1:
                    lr *= 0.5
                elif it % 100 == 0 and it > 0:
                    lr = max(min_lr, lr * 0.9)
                
                prev_obj = obj
            
            if best_rank < target_rank:
                break
        
        if best_rank < target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
