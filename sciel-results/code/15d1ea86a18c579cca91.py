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
    time_limit = 140.0  # Leave 10 seconds margin before 150s kill
    
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
    
    # Build the target tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Try Kronecker product approach:
    # The 3x5x5 tensor can be viewed as a Kronecker product structure.
    # Specifically, if we decompose the 3x5x5 tensor as a sum of rank-1 terms,
    # we can try to find a low-rank decomposition.
    
    # Let's try a random search approach for small ranks
    # Start with rank 25 (the trivial lower bound) and see if we can find a solution
    
    # Actually, let's try a more systematic approach using optimization.
    # We'll parameterize U, V, W and minimize the error.
    
    def try_rank(rank):
        """Try to find a rank-`rank` decomposition using optimization."""
        if time.time() - start_time > time_limit - 20:
            return None
        
        # Initialize with random values
        U_init = np.random.randn(n * m, rank) * 0.1
        V_init = np.random.randn(m * p, rank) * 0.1
        W_init = np.random.randn(p * n, rank) * 0.1
        
        def objective(params):
            if time.time() - start_time > time_limit - 10:
                return 1e10
            U = params[:n * m * rank].reshape(n * m, rank)
            V = params[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
            W = params[n * m * rank + m * p * rank:].reshape(p * n, rank)
            
            # Compute the error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                        error += (val - 1.0) ** 2
            return error
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.success or result.fun < 1e-6:
                U = result.x[:n * m * rank].reshape(n * m, rank)
                V = result.x[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
                W = result.x[n * m * rank + m * p * rank:].reshape(p * n, rank)
                
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Check if the rounded solution is exact
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_rounded[i * m + j] * V_rounded[j * p + k] * W_rounded[k * n + i])
                            error += (val - 1.0) ** 2
                
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
    # But only for a limited number of ranks to stay within time budget
    max_ranks_to_try = 5
    for rank in range(max(n * m, m * p, p * n), best_rank + 1, max(1, (best_rank - max(n * m, m * p, p * n)) // max_ranks_to_try)):
        if time.time() - start_time > time_limit - 30:
            break
        
        result = try_rank(rank)
        if result is not None:
            best_result = result
            best_rank = rank
            if best_rank == max(n * m, m * p, p * n):
                break
    
    # Check time
    if time.time() - start_time > time_limit:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
