"""Matrix multiplication 2x3 by 3x3, rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W, n, m, p):
    """Check U,V,W of given shapes reconstruct the matmul tensor exactly."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    if recon.shape != T.shape:
        return False
    return np.allclose(recon, T, atol=1e-9)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin: 10s before 150s kill

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to find a lower rank decomposition
    # For 2x3x3 matrix multiplication, the known lower bound is 15 (from asymptotic results)
    # The trivial upper bound is 18 (schoolbook)
    # We'll attempt to find a rank-15 or rank-16 decomposition using numerical optimization

    def try_rank(r):
        """Try to find a rank-r decomposition using numerical optimization."""
        if time.time() > deadline:
            return None
            
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Target tensor
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0

        # Flatten the target tensor
        T_flat = T.flatten()
        
        # Objective function: minimize ||U ⊙ V ⊙ W - T||^2
        def objective(params):
            U_flat, V_flat, W_flat = params[:n*m*r], params[n*m*r:n*m*r+m*p*r], params[n*m*r+m*p*r:]
            U_new = U_flat.reshape(n * m, r)
            V_new = V_flat.reshape(m * p, r)
            W_new = W_flat.reshape(p * n, r)
            recon = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
            diff = recon.flatten() - T_flat
            return np.sum(diff ** 2)
        
        # Initial parameters
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use gradient descent with momentum
        lr = 0.01
        momentum = 0.9
        velocity = np.zeros_like(params)
        best_params = params.copy()
        best_obj = objective(params)
        
        for step in range(500):
            if time.time() > deadline:
                break
                
            # Compute gradient numerically
            grad = np.zeros_like(params)
            eps = 1e-5
            for idx in range(len(params)):
                params_plus = params.copy()
                params_plus[idx] += eps
                grad[idx] = (objective(params_plus) - best_obj) / eps
            
            # Update parameters
            velocity = momentum * velocity - lr * grad
            params = params + velocity
            
            current_obj = objective(params)
            if current_obj < best_obj:
                best_obj = current_obj
                best_params = params.copy()
            
            # Check convergence
            if best_obj < 1e-10:
                break
        
        # Reconstruct from best_params
        U_best = best_params[:n*m*r].reshape(n * m, r)
        V_best = best_params[n*m*r:n*m*r+m*p*r].reshape(m * p, r)
        W_best = best_params[n*m*r+m*p*r:].reshape(p * n, r)
        
        # Verify
        if _verify(U_best, V_best, W_best, n, m, p):
            # Convert to lists with integer/half-integer values if possible
            def to_integer_list(arr):
                return [[float(x) for x in row] for row in arr]
            return {"U": to_integer_list(U_best), "V": to_integer_list(V_best), "W": to_integer_list(W_best)}
        
        return None

    # Try ranks from 15 to 17 (skip 18 since schoolbook is trivial)
    for r in range(15, 18):
        result = try_rank(r)
        if result is not None:
            return result
    
    # If no better rank found, return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
