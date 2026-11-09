"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import minimize

def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Fast verification using numpy."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_t[i, j, :] * V_t[j, k, :] * W_t[k, i, :]
                s = np.sum(prod)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W, rank

def _optimize_decomposition(n, m, p, rank, max_iter=1000, tol=1e-8, time_budget=20.0):
    """Use gradient-based optimization to find a rank-rank decomposition."""
    # Target: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # We minimize sum over all (i,j,k) of (sum_c ... - 1)^2
    
    def objective(x):
        U_flat = x[:n*m*rank]
        V_flat = x[n*m*rank : n*m*rank + m*p*rank]
        W_flat = x[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
        
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i,j,:] * V[j,k,:] * W[k,i,:])
                    err += (s - 1.0) ** 2
        return err
    
    def gradient(x):
        U_flat = x[:n*m*rank]
        V_flat = x[n*m*rank : n*m*rank + m*p*rank]
        W_flat = x[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
        
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i,j,:] * V[j,k,:] * W[k,i,:])
                    ds = 2.0 * (s - 1.0)
                    dU[i,j,:] += ds * V[j,k,:] * W[k,i,:]
                    dV[j,k,:] += ds * U[i,j,:] * W[k,i,:]
                    dW[k,i,:] += ds * U[i,j,:] * V[j,k,:]
        
        return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    
    # Random initialization
    np.random.seed(42)
    U_init = np.random.randn(n, m, rank) * 0.5
    V_init = np.random.randn(m, p, rank) * 0.5
    W_init = np.random.randn(p, n, rank) * 0.5
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    start_time = time.time()
    
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
    
    elapsed = time.time() - start_time
    
    if result.fun < tol:
        U_flat = result.x[:n*m*rank]
        V_flat = result.x[n*m*rank : n*m*rank + m*p*rank]
        W_flat = result.x[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        # Convert to list of lists
        U_list = [[float(U[i,c]) for c in range(rank)] for i in range(n*m)]
        V_list = [[float(V[i,c]) for c in range(rank)] for i in range(m*p)]
        W_list = [[float(W[i,c]) for c in range(rank)] for i in range(p*n)]
        
        # Verify
        if _verify_fast(U_list, V_list, W_list, n, m, p, tol=1e-8):
            return U_list, V_list, W_list, result.fun, elapsed
        else:
            return None, None, None, result.fun, elapsed
    else:
        return None, None, None, result.fun, elapsed

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # 130s to leave margin
    
    # Start with schoolbook
    U, V, W, best_rank = _schoolbook(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    
    # Try optimization for various ranks
    # Theoretical lower bound is max(n*m, m*p, p*n) = 25
    # Known results suggest rank around 40-50 for 3x5x5
    # Try ranks from 25 upward
    
    for target_rank in range(25, best_rank):
        if time.time() > deadline - 15:
            break
        
        # Try a few random restarts
        found = False
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            # Use different seeds for different restarts
            np.random.seed(42 + restart * 100)
            
            U_r, V_r, W_r, obj_val, elapsed = _optimize_decomposition(
                n, m, p, target_rank, max_iter=500, tol=1e-10, time_budget=25.0
            )
            
            if U_r is not None:
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                found = True
                break
        
        if found:
            # Try to go lower
            continue
    
    return best_result
# EVOLVE-BLOCK-END
