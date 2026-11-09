"""Matrix multiplication 3x4x8 rank-minimal tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np
from scipy.optimize import minimize
from fractions import Fraction

def _trivial_decomp(n, m, p):
    """Return the trivial rank n*m*p decomposition as numpy arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                r = i * (m * p) + k * p + j
                U[i * m + k, r] = 1.0
                V[k * p + j, r] = 1.0
                W[j * n + i, r] = 1.0
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using numpy einsum."""
    rank = U.shape[1]
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # T[i,k,j] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
    T = np.einsum('ikr,kjr,jir->ijk', U_3d, V_3d, W_3d, optimize=True)
    
    return np.allclose(T, 1.0, atol=tol)

def _is_half_integer(x):
    """Check if x is an integer or half-integer (within tolerance)."""
    if x is None or np.isnan(x):
        return False
    x = float(x)
    return abs(x * 2 - round(x * 2)) < 1e-6

def _snap_to_half_integer(x):
    """Snap x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _snap_matrix(mat):
    """Snap all entries to nearest half-integer."""
    return np.array([[_snap_to_half_integer(x) for x in row] for row in mat])

def _reduce_rank(U, V, W, n, m, p, tol=1e-8):
    """Try to reduce the rank by removing redundant terms."""
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    changed = True
    while changed:
        changed = False
        for r in range(U.shape[1]):
            U_no_r = np.delete(U, r, axis=1)
            V_no_r = np.delete(V, r, axis=1)
            W_no_r = np.delete(W, r, axis=1)
            if _verify_fast(U_no_r, V_no_r, W_no_r, n, m, p, tol):
                U = U_no_r
                V = V_no_r
                W = W_no_r
                changed = True
                break # Restart loop
    return U, V, W, U.shape[1]

def _objective(x, n, m, p):
    rank = int(len(x) / (n*m + m*p + p*n))
    U_flat = x[:n*m*rank].reshape(n*m, rank)
    V_flat = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
    W_flat = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
    
    U_3d = U_flat.reshape(n, m, rank)
    V_3d = V_flat.reshape(m, p, rank)
    W_3d = W_flat.reshape(p, n, rank)
    
    T = np.einsum('ikr,kjr,jir->ijk', U_3d, V_3d, W_3d, optimize=True)
    return np.sum((T - 1.0)**2)

def _optimize_decomposition(n, m, p, rank, time_budget):
    t_start = time.time()
    # Random initialization
    U = np.random.randn(n*m, rank) * 0.5
    V = np.random.randn(m*p, rank) * 0.5
    W = np.random.randn(p*n, rank) * 0.5
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    best_loss = float('inf')
    best_x = x0
    
    while time.time() - t_start < time_budget:
        try:
            res = minimize(_objective, x0, args=(n, m, p), method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7})
            if res.fun < best_loss:
                best_loss = res.fun
                best_x = res.x
        except Exception:
            pass
        
        # Restart
        x0 = best_x + np.random.randn(len(best_x)) * 0.1
        
        if best_loss < 1e-8:
            break
            
    if best_loss > 1e-8:
        return None, None, None, rank
        
    U_opt = best_x[:n*m*rank].reshape(n*m, rank)
    V_opt = best_x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
    W_opt = best_x[n*m*rank + m*p*rank:].reshape(p*n, rank)
    
    # Snap to half-integers
    U_snap = _snap_matrix(U_opt)
    V_snap = _snap_matrix(V_opt)
    W_snap = _snap_matrix(W_opt)
    
    if _verify_fast(U_snap, V_snap, W_snap, n, m, p, tol=1e-8):
        U_final, V_final, W_final, r_final = _reduce_rank(U_snap, V_snap, W_snap, n, m, p)
        return U_final, V_final, W_final, r_final
        
    return None, None, None, rank

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 145.0  # 5s safety margin
    
    # Start with the trivial decomposition
    U, V, W = _trivial_decomp(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = n * m * p
    
    # Try to find better solutions
    # Lower bound is max(n*m, m*p, n*p) = 32
    lower_bound = max(n*m, m*p, n*p)
    
    for rank in range(lower_bound, best_rank + 1):
        if time.time() > deadline - 10:
            break
            
        # Allocate time per rank
        # Give more time to lower ranks
        time_budget = min(5.0, deadline - time.time() - 5)
        if time_budget < 1:
            break
            
        U_opt, V_opt, W_opt, r_opt = _optimize_decomposition(n, m, p, rank, time_budget)
        
        if U_opt is not None and r_opt < best_rank:
            best_rank = r_opt
            best_U, best_V, best_W = U_opt, V_opt, W_opt
            
            # If we found a solution close to the lower bound, we can stop
            if rank <= lower_bound + 2:
                break
                
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
