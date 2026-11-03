"""Structural tensor decomposition for 4x4 * 4x5 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    def compute_loss(U, V, W):
        """Compute the loss for a given decomposition.
        U: (n*m, r), V: (m*p, r), W: (p*n, r)
        For each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1.
        """
        r = U.shape[1]
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    loss += (val - 1.0) ** 2
        return loss

    def try_rank(r, num_restarts=5, max_iter=500):
        """Try to find a rank-r decomposition using gradient descent."""
        np.random.seed(42)  # Fixed seed for reproducibility
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize with random values
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Use L-BFGS or simple gradient descent
            # We'll use scipy's minimize if available, otherwise manual
            try:
                from scipy.optimize import minimize
            except ImportError:
                # Fallback to manual gradient descent
                lr = 0.01
                prev_loss = float('inf')
                for it in range(max_iter):
                    if time.time() > deadline:
                        return None
                    
                    # Compute loss
                    loss = compute_loss(U, V, W)
                    if abs(loss) < 1e-10:
                        return U, V, W
                    
                    # Compute gradients numerically
                    eps = 1e-7
                    grad_U = np.zeros_like(U)
                    grad_V = np.zeros_like(V)
                    grad_W = np.zeros_like(W)
                    
                    # This is too slow for full numerical gradients
                    # Use analytic gradients instead
                    break
                
                return None
            
            # Use scipy minimize
            def objective(x):
                U_x = x[:n*m*r].reshape(n*m, r)
                V_x = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_x = x[n*m*r + m*p*r:].reshape(p*n, r)
                return compute_loss(U_x, V_x, W_x)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.fun < 1e-8:
                U_opt = result.x[:n*m*r].reshape(n*m, r)
                V_opt = result.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_opt = result.x[n*m*r + m*p*r:].reshape(p*n, r)
                return U_opt, V_opt, W_opt
        
        return None

    # Try ranks from low to high
    # Known lower bound: for 4x4*4x5, the tensor rank is at least 4*4*5/4 = 20 (rough)
    # But realistically, we expect something between 30 and 50
    best_result = None
    for r in range(25, 55):
        if time.time() > deadline:
            break
        
        result = try_rank(r, num_restarts=3, max_iter=200)
        if result is not None:
            U, V, W = result
            # Verify
            loss = compute_loss(U, V, W)
            if loss < 1e-8:
                best_result = (U, V, W, r)
                break
    
    if best_result is None:
        # Fallback to schoolbook
        rank = 80
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
    else:
        U, V, W, rank = best_result

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
