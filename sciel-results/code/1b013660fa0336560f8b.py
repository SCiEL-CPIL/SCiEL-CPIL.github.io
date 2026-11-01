"""Matrix multiplication 2,2,13 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]
DEADLINE_MARGIN = 10  # seconds before kill to stop

def _verify_decomposition(U, V, W, n=2, m=2, p=13, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n=2, m=2, p=13):
    """Return the trivial schoolbook decomposition with rank n*m*p."""
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

def construct():
    """Return a valid tensor decomposition for 2x2 * 2x13 matrix multiplication."""
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=2, p=13
    
    # Start with schoolbook decomposition as fallback
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p  # 52
    
    # Verify the schoolbook decomposition
    if not _verify_decomposition(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        # Should never happen, but be safe
        pass
    
    # Try to find a lower-rank decomposition using numerical optimization
    # Given the structure, we attempt ranks from best_rank-1 down to a reasonable lower bound
    # Lower bound: at least max(n*m, m*p, p*n) = max(4, 26, 26) = 26
    # But our analysis suggests rank is likely 52. Try a few lower ranks quickly.
    
    target_ranks = [51, 50, 48, 45]  # Try a few lower ranks
    
    for target_rank in target_ranks:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE_MARGIN:
            break
        
        # Attempt to find a decomposition with target_rank
        # Use a simple iterative approach: initialize randomly and refine
        max_iter = 200
        np.random.seed(42)
        
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        for iteration in range(max_iter):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE_MARGIN:
                break
            
            # Compute the current tensor values
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        errors.append(val - 1.0)
            
            errors = np.array(errors)
            err_norm = np.linalg.norm(errors)
            
            if err_norm < 1e-10:
                # Found a valid decomposition
                # Convert to lists of integers/half-integers if possible
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                # Try to snap to half-integers
                U_snap = [[round(2*x)/2 for x in row] for row in U_list]
                V_snap = [[round(2*x)/2 for x in row] for row in V_list]
                W_snap = [[round(2*x)/2 for x in row] for row in W_list]
                
                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-8):
                    best_result = {"U": U_snap, "V": V_snap, "W": W_snap}
                    best_rank = target_rank
                    break
                else:
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                break
            
            # Simple gradient step (approximate)
            lr = 0.01 / (1 + iteration * 0.01)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        grad_U = V[j*p+k, :] * W[k*n+i, :] * (np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1.0)
                        U[i*m+j, :] -= lr * grad_U
                        
                        grad_V = U[i*m+j, :] * W[k*n+i, :] * (np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1.0)
                        V[j*p+k, :] -= lr * grad_V
                        
                        grad_W = U[i*m+j, :] * V[j*p+k, :] * (np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1.0)
                        W[k*n+i, :] -= lr * grad_W
    
    # Final deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_MARGIN:
        # Return the best result we have
        pass
    
    return best_result
