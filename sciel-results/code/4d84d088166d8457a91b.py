import time
import numpy as np
import scipy.optimize as optimize

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=3, m=3, p=3):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _round_to_half(x):
    return np.round(2 * x) / 2.0

def _find_decomposition(r, n=3, m=3, p=3, time_budget=10.0, restarts=5):
    """
    Try to find a rank-r decomposition using a robust optimization approach.
    """
    start = time.time()
    
    # Objective function: minimize sum of squared errors
    def objective(params):
        if time.time() - start > time_budget:
            return 1e10
        U = params[0:n*m*r].reshape(n*m, r)
        V = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
        W = params[n*m*r+m*p*r:].reshape(p*n, r)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err += (val - 1.0) ** 2
        return err

    best_err = np.inf
    best_params = None
    
    for restart in range(restarts):
        if time.time() - start > time_budget:
            break
            
        np.random.seed(42 + restart)
        # Initialize with small random values
        init_params = np.random.randn(n*m*r + m*p*r + p*n*r) * 0.5
        
        # Use differential evolution for global search
        bounds = [(-10, 10)] * (n*m*r + m*p*r + p*n*r)
        
        try:
            res = optimize.differential_evolution(
                objective, 
                bounds, 
                maxiter=100, 
                tol=1e-6,
                popsize=10,
                seed=42+restart
            )
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
        except Exception as e:
            pass

    if best_params is None or best_err > 1e-6:
        return None

    # Round to half-integers
    U = best_params[0:n*m*r].reshape(n*m, r)
    V = best_params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
    W = best_params[n*m*r+m*p*r:].reshape(p*n, r)
    
    U_rounded = _round_to_half(U)
    V_rounded = _round_to_half(V)
    W_rounded = _round_to_half(W)
    
    if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
        return U_rounded, V_rounded, W_rounded
    
    return None

def construct():
    start_time = time.time()
    timeout = 120.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Guaranteed rank-27 decomposition
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try rank 23
    if time.time() - start_time < timeout - 10:
        result = _find_decomposition(23, n, m, p, time_budget=timeout - 15, restarts=3)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = 23
            return best_result
            
    return best_result
# EVOLVE-BLOCK-END
