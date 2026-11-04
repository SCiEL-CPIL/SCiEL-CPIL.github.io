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
    nm, mp, pn = n * m, m * p, p * n
    
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
    T_target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0

    def compute_error(U, V, W):
        """Compute the sum of squared errors between the reconstructed tensor and the target."""
        T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.sum((T_rec - T_target) ** 2)

    def try_rank(rank, max_iters=1000, ftol=1e-10, gtol=1e-6):
        """Try to find a rank-`rank` decomposition using optimization."""
        if time.time() - start_time > time_limit - 10:
            return None
        
        # Multiple restarts with different initializations
        n_restarts = 5
        best_local_error = float('inf')
        best_local_params = None
        
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit - 15:
                break
            
            # Initialize with random values
            if restart == 0:
                # Start with small random values
                U_init = np.random.randn(nm, rank) * 0.1
                V_init = np.random.randn(mp, rank) * 0.1
                W_init = np.random.randn(pn, rank) * 0.1
            elif restart == 1:
                # Start with larger random values
                U_init = np.random.randn(nm, rank) * 0.5
                V_init = np.random.randn(mp, rank) * 0.5
                W_init = np.random.randn(pn, rank) * 0.5
            elif restart == 2:
                # Start with structured initialization
                U_init = np.zeros((nm, rank))
                V_init = np.zeros((mp, rank))
                W_init = np.zeros((pn, rank))
                for r in range(rank):
                    # Try to assign some structure
                    for idx in range(nm):
                        if (idx // m) % rank == r:
                            U_init[idx, r] = 1.0
                    for idx in range(mp):
                        if (idx // p) % rank == r:
                            V_init[idx, r] = 1.0
                    for idx in range(pn):
                        if (idx % n) % rank == r:
                            W_init[idx, r] = 1.0
                # Add small noise
                U_init += np.random.randn(nm, rank) * 0.01
                V_init += np.random.randn(mp, rank) * 0.01
                W_init += np.random.randn(pn, rank) * 0.01
            elif restart == 3:
                # Start with even larger random values
                U_init = np.random.randn(nm, rank) * 1.0
                V_init = np.random.randn(mp, rank) * 1.0
                W_init = np.random.randn(pn, rank) * 1.0
            else:
                # Start with very small random values
                U_init = np.random.randn(nm, rank) * 0.01
                V_init = np.random.randn(mp, rank) * 0.01
                W_init = np.random.randn(pn, rank) * 0.01
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(params):
                if time.time() - start_time > time_limit - 5:
                    return 1e10
                U = params[:nm * rank].reshape(nm, rank)
                V = params[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
                W = params[nm * rank + mp * rank:].reshape(pn, rank)
                T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
                error = np.sum((T_rec - T_target) ** 2)
                return error
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                options={'maxiter': max_iters, 'ftol': ftol, 'gtol': gtol, 'maxls': 50})
                
                if result.fun < best_local_error:
                    best_local_error = result.fun
                    best_local_params = result.x.copy()
                
                if result.fun < 1e-6:
                    U = result.x[:nm * rank].reshape(nm, rank)
                    V = result.x[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
                    W = result.x[nm * rank + mp * rank:].reshape(pn, rank)
                    
                    # Round to nearest half-integer
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Check if the rounded solution is exact
                    error = compute_error(U_rounded, V_rounded, W_rounded)
                    
                    if error < 1e-9:
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
            except Exception:
                pass
        
        # If we didn't find an exact solution, try to polish the best local solution
        if best_local_params is not None and best_local_error < 1e-4:
            try:
                U = best_local_params[:nm * rank].reshape(nm, rank)
                V = best_local_params[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
                W = best_local_params[nm * rank + mp * rank:].reshape(pn, rank)
                
                # Try rounding to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                error = compute_error(U_rounded, V_rounded, W_rounded)
                
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
    lower_bound = max(nm, mp, pn)
    
    max_ranks_to_try = 15  # Limit number of rank attempts to stay within time
    for rank in range(lower_bound, min(lower_bound + max_ranks_to_try, best_rank + 1)):
        if time.time() - start_time > time_limit - 20:
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
