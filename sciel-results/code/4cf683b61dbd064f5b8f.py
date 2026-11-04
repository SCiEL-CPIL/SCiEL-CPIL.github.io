import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
        
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check dimensions
    if U_np.shape != (n*m, rank) or V_np.shape != (m*p, rank) or W_np.shape != (p*n, rank):
        return False
        
    # Compute the contraction
    # T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    U_reshaped = U_np.reshape(n, m, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_reshaped[i, j, c] * V_reshaped[j, k, c] * W_reshaped[k, i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _snap_to_half(x):
    """Snap a value to the nearest multiple of 0.5."""
    return np.round(x * 2) / 2

def _error(U, V, W, n, m, p):
    """Calculate the error of the decomposition."""
    U_np = np.array(U, dtype=float).reshape(n, m, -1)
    V_np = np.array(V, dtype=float).reshape(m, p, -1)
    W_np = np.array(W, dtype=float).reshape(p, n, -1)
    
    rank = U_np.shape[2]
    # Compute T
    T = np.zeros((n, m, p))
    for c in range(rank):
        T += U_np[:, :, c][:, :, np.newaxis] * V_np[:, :, c][np.newaxis, :, :] * W_np[:, :, c][np.newaxis, np.newaxis, :]
        
    # Target is all 1s
    target = np.ones((n, m, p))
    return np.sum((T - target) ** 2)

def construct():
    start_time = time.time()
    deadline = 110.0
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
                
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Try to find a lower rank decomposition
    # Known lower bound is 21. We try ranks from 21 upwards.
    for rank in range(21, 36):
        if time.time() > deadline:
            break
            
        # Multiple restarts for this rank
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize with random values
            # Use small integers to help with snapping later
            U = np.random.randint(-1, 2, (n * m, rank)) * 0.5
            V = np.random.randint(-1, 2, (m * p, rank)) * 0.5
            W = np.random.randint(-1, 2, (p * n, rank)) * 0.5
            
            # Flatten for optimization
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_flat = x[:n*m*rank]
                V_flat = x[n*m*rank : n*m*rank + m*p*rank]
                W_flat = x[n*m*rank + m*p*rank :]
                
                U = U_flat.reshape(n*m, rank)
                V = V_flat.reshape(m*p, rank)
                W = W_flat.reshape(p*n, rank)
                
                return _error(U, V, W, n, m, p)
            
            # Use Nelder-Mead for local search
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
                
                if res.fun < 1e-6:
                    # Snap to nearest half-integer
                    U_opt = res.x[:n*m*rank].reshape(n*m, rank)
                    V_opt = res.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_opt = res.x[n*m*rank + m*p*rank :].reshape(p*n, rank)
                    
                    U_snap = _snap_to_half(U_opt)
                    V_snap = _snap_to_half(V_opt)
                    W_snap = _snap_to_half(W_opt)
                    
                    # Convert to lists for return format
                    U_list = U_snap.tolist()
                    V_list = V_snap.tolist()
                    W_list = W_snap.tolist()
                    
                    if _verify(U_list, V_list, W_list, n, m, p):
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
            except Exception:
                pass

    return best_result
