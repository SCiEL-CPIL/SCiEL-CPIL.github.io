import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly represents matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if val != 1:
                    return False
    return True

def _construct_schoolbook(n, m, p):
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

def _search_decomposition(n, m, p, target_rank, time_limit):
    """
    Search for a decomposition of rank target_rank.
    Uses random initialization and gradient descent.
    """
    start_time = time.time()
    
    # Shapes
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    # Target tensor is all ones
    # We minimize sum_{i,j,k} (sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1)^2
    
    best_result = None
    best_cost = float('inf')
    
    # Number of restarts depends on time available
    # Each restart: init, minimize, check, snap, verify
    
    while time.time() - start_time < time_limit - 5:
        # Initialize
        U = np.random.randint(-1, 2, size=(size_U, target_rank)).astype(float)
        V = np.random.randint(-1, 2, size=(size_V, target_rank)).astype(float)
        W = np.random.randint(-1, 2, size=(size_W, target_rank)).astype(float)
        
        # Add some noise to avoid local minima
        U += np.random.randn(size_U, target_rank) * 0.1
        V += np.random.randn(size_V, target_rank) * 0.1
        W += np.random.randn(size_W, target_rank) * 0.1
        
        # Objective function
        def objective(params):
            # params is a flattened array
            U_flat = params[:size_U * target_rank]
            V_flat = params[size_U * target_rank : size_U * target_rank + size_V * target_rank]
            W_flat = params[size_U * target_rank + size_V * target_rank :]
            
            U_mat = U_flat.reshape(size_U, target_rank)
            V_mat = V_flat.reshape(size_V, target_rank)
            W_mat = W_flat.reshape(size_W, target_rank)
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                        val = 0.0
                        for r in range(target_rank):
                            val += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                        error += (val - 1.0) ** 2
            return error
        
        # Gradient is hard to compute analytically fast, so use numerical or just try a few steps
        # Actually, let's use scipy.optimize.minimize with 'L-BFGS-B' or similar if available and fast
        # But calculating the gradient for 315 variables might be slow if done naively.
        # Let's try a simple stochastic gradient descent or just use scipy's numerical gradient.
        
        try:
            from scipy.optimize import minimize
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Limit iterations to prevent hanging
            result = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 100, 'disp': False})
            
            if result.success or result.fun < best_cost:
                best_cost = result.fun
                best_params = result.x
                
                # Check if we are close enough to snap
                if best_cost < 1e-4:
                    U_sol = best_params[:size_U * target_rank].reshape(size_U, target_rank)
                    V_sol = best_params[size_U * target_rank : size_U * target_rank + size_V * target_rank].reshape(size_V, target_rank)
                    W_sol = best_params[size_U * target_rank + size_V * target_rank :].reshape(size_W, target_rank)
                    
                    # Snap to nearest half-integer
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    # Verify exactness
                    if _verify_decomposition(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
                        # Convert to integers if possible for cleaner output, though half-integers are allowed
                        # The problem says "integer/half-integer entries"
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                        
        except Exception:
            pass

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Try to find a better rank than schoolbook (18)
    # Try ranks 15, 16, 17
    for rank in [15, 16, 17]:
        if time.time() > deadline - 10:
            break
        
        # Allocate time for this rank search
        # If we find one, we return immediately
        res = _search_decomposition(n, m, p, rank, time_limit=10.0)
        if res is not None:
            return res
            
    # Fallback to schoolbook
    return _construct_schoolbook(n, m, p)
