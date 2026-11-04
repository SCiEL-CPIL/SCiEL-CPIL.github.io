"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    start_time = time.time()
    time_limit = 145.0  # Leave 5 seconds margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Construct the target tensor T
    # T is a 3D array of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # All other entries are 0.
    T_target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0

    def try_rank(rank):
        """Try to find a rank-`rank` decomposition using optimization."""
        if time.time() - start_time > time_limit - 10:
            return None
        
        # Initialize with random values
        # Use small values to start near zero, or random normal
        U_init = np.random.randn(n * m, rank) * 0.1
        V_init = np.random.randn(m * p, rank) * 0.1
        W_init = np.random.randn(p * n, rank) * 0.1
        
        def objective(params):
            if time.time() - start_time > time_limit - 5:
                return 1e10
            U = params[:n * m * rank].reshape(n * m, rank)
            V = params[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
            W = params[n * m * rank + m * p * rank:].reshape(p * n, rank)
            
            # Compute T_rec[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
            # Efficiently:
            # T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
            T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
            
            # Error is sum of squares of (T_rec - T_target)
            error = np.sum((T_rec - T_target) ** 2)
            return error
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            # Use L-BFGS-B for large-scale optimization
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            
            if result.fun < 1e-6:
                U = result.x[:n * m * rank].reshape(n * m, rank)
                V = result.x[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
                W = result.x[n * m * rank + m * p * rank:].reshape(p * n, rank)
                
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Check if the rounded solution is exact
                T_rec_rounded = np.einsum('ur,vr,wr->uvw', U_rounded, V_rounded, W_rounded)
                error = np.sum((T_rec_rounded - T_target) ** 2)
                
                if error < 1e-9:
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
        except Exception:
            pass
        
        return None
    
    # Try ranks from the lower bound up to the schoolbook rank
    lower_bound = max(n * m, m * p, p * n)
    
    # We prioritize finding a solution. 
    # Start from lower bound. If it fails, increment.
    # To save time, we might skip some ranks if optimization is slow, 
    # but for now, let's try consecutive ranks with a strict time check.
    
    max_ranks_to_try = 10 # Limit number of rank attempts to stay within time
    for rank in range(lower_bound, best_rank + 1):
        if time.time() - start_time > time_limit - 15:
            break
        
        result = try_rank(rank)
        if result is not None:
            best_result = result
            best_rank = rank
            if best_rank == lower_bound:
                break
    
    # Check time
    if time.time() - start_time > time_limit:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
