"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank decomposition using optimization
    # Start with naive rank and try to reduce
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Start with naive decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    
    # Try to reduce rank by attempting to eliminate terms
    # For each term, try to see if it can be removed or merged
    
    # Simple heuristic: try random perturbations and see if we can reduce rank
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find a lower rank by using random search
    np.random.seed(42)
    
    for target_rank in range(rank - 1, max(10, rank - 10), -1):
        if time.time() > deadline - 30:
            break
        
        # Try random initialization for the lower rank
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
            
            U_try = np.zeros((n * m, target_rank))
            V_try = np.zeros((m * p, target_rank))
            W_try = np.zeros((p * n, target_rank))
            
            # Initialize with random values
            U_try[:, :] = np.random.randn(n * m, target_rank) * 0.5
            V_try[:, :] = np.random.randn(m * p, target_rank) * 0.5
            W_try[:, :] = np.random.randn(p * n, target_rank) * 0.5
            
            # Use gradient descent to minimize the error
            lr = 0.01
            for step in range(1000):
                if time.time() > deadline - 20:
                    break
                
                # Compute gradients
                grad_U = np.zeros_like(U_try)
                grad_V = np.zeros_like(V_try)
                grad_W = np.zeros_like(W_try)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            # Current value
                            val = np.sum(U_try[a, :] * V_try[b, :] * W_try[c, :])
                            err = val - 1.0
                            # Gradients
                            for r in range(target_rank):
                                grad_U[a, r] += err * V_try[b, r] * W_try[c, r]
                                grad_V[b, r] += err * U_try[a, r] * W_try[c, r]
                                grad_W[c, r] += err * U_try[a, r] * V_try[b, r]
                
                # Update
                U_try -= lr * grad_U
                V_try -= lr * grad_V
                W_try -= lr * grad_W
                
                # Check if converged
                if np.max(np.abs(grad_U)) < 1e-6 and np.max(np.abs(grad_V)) < 1e-6 and np.max(np.abs(grad_W)) < 1e-6:
                    break
            
            # Check if we found a valid decomposition
            U_list = U_try.tolist()
            V_list = V_try.tolist()
            W_list = W_try.tolist()
            
            if check_decomposition(U_list, V_list, W_list, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U_list, V_list, W_list
                break
    
    # Convert to lists if needed
    if isinstance(best_U, np.ndarray):
        best_U = best_U.tolist()
    if isinstance(best_V, np.ndarray):
        best_V = best_V.tolist()
    if isinstance(best_W, np.ndarray):
        best_W = best_W.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
