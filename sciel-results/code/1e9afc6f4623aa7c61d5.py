"""Matrix multiplication 3x4x7: rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 7]
DEADLINE_MARGIN = 10.0  # seconds before the 150s wall-clock limit

def _verify(U, V, W, n=3, m=4, p=7, tol=1e-10):
    """Verify that U, V, W give an exact decomposition of matrix mult tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_3d[i, j, :], np.dot(V_3d[j, k, :], W_3d[k, i, :]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    return max_err < tol

def _verify_fast(U, V, W, n=3, m=4, p=7, tol=1e-10):
    """Faster verification using numpy broadcasting."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(n, m, rank)
    V_arr = np.array(V, dtype=float).reshape(m, p, rank)
    W_arr = np.array(W, dtype=float).reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol

def _schoolbook(n=3, m=4, p=7):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _objective(params, n, m, p, rank):
    """Objective function for optimization: sum of squared errors."""
    U_flat = params[:n * m * rank]
    V_flat = params[n * m * rank : n * m * rank + m * p * rank]
    W_flat = params[n * m * rank + m * p * rank : ]
    
    U = U_flat.reshape(n, m, rank)
    V = V_flat.reshape(m, p, rank)
    W = W_flat.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    err = np.sum((T - 1.0) ** 2)
    return err

def _try_reduce_rank_opt(n, m, p=7, target_rank=None, seed=42, time_budget=120.0):
    """
    Try to find a decomposition of rank < n*m*p using scipy.optimize.
    """
    if target_rank is None:
        target_rank = n * m * p - 1
    
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    # Scale appropriately
    scale = 1.0 / (n * m * p) ** 0.5
    
    best_err = float('inf')
    best_params = None
    
    # Try a few random restarts
    max_restarts = 5
    
    for r in range(max_restarts):
        if time.time() - start_time > time_budget - DEADLINE_MARGIN:
            break
            
        # Random initialization
        U_init = rng.randn(n * m, target_rank) * scale
        V_init = rng.randn(m * p, target_rank) * scale
        W_init = rng.randn(p * n, target_rank) * scale
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Use L-BFGS-B or Nelder-Mead
        # Nelder-Mead is derivative-free but slow in high dimensions.
        # L-BFGS-B is better but needs gradients.
        # Let's try Nelder-Mead for simplicity, or just a few steps of gradient descent if we implement gradients.
        # Given the dimensionality (84*3 = 252 vars), Nelder-Mead might be too slow.
        
        # Let's try a simple gradient descent with adaptive step size or just use the provided `minimize` with 'L-BFGS-B' and numerical gradients.
        
        res = minimize(_objective, params_init, args=(n, m, p, target_rank), method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if res.success or res.fun < best_err:
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
        
        if best_err < 1e-10:
            break
            
    if best_err < 1e-10 and best_params is not None:
        U_flat = best_params[:n * m * target_rank]
        V_flat = best_params[n * m * target_rank : n * m * target_rank + m * p * target_rank]
        W_flat = best_params[n * m * target_rank + m * p * target_rank : ]
        
        U = U_flat.reshape(n * m, target_rank).tolist()
        V = V_flat.reshape(m * p, target_rank).tolist()
        W = W_flat.reshape(p * n, target_rank).tolist()
        
        return U, V, W, best_err
        
    return None, None, None, best_err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    current_rank = n * m * p
    
    # Try to reduce rank
    # Try various target ranks from current_rank - 1 downwards
    # We limit the number of attempts to ensure we finish in time
    for target_rank in range(current_rank - 1, max(n * m, m * p, p * n) - 1, -1):
        if time.time() - start_time > 120.0 - DEADLINE_MARGIN:
            break
            
        # Only try a few ranks to avoid spending all time on one difficult rank
        if target_rank > current_rank - 5:
             U, V, W, err = _try_reduce_rank_opt(n, m, p, target_rank, seed=42, time_budget=120.0 - (time.time() - start_time))
             
             if U is not None and err < 1e-10:
                 result = {"U": U, "V": V, "W": W}
                 current_rank = target_rank
                 # Continue trying to reduce further
             else:
                 # If we fail to reduce, we might just stick with schoolbook or try a slightly lower rank later
                 # But for now, if we can't find rank 83, we return 84.
                 # We can try rank 82, 81... but it's risky.
                 # Let's just return the best found so far.
                 pass
    
    return result

# For testing
if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
    print(f"Valid: {_verify_fast(result['U'], result['V'], result['W'])}")
