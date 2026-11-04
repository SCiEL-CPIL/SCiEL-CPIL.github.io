"""Improved program for 3x3 * 3x8 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if U, V, W form a valid tensor decomposition."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Compute the tensor from the decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # The target tensor is 1 if (i,j,k) is a valid index triple, else 0
    # For matrix multiplication C = A*B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = delta (i.e., 1 for all valid i,j,k since it's the multiplication tensor)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                if max_err > tol:
                    return False, max_err
    return True, max_err

def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Faster verification using tensor operations."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    T = np.zeros((n, m, p))
    for c in range(rank):
        T += Ur[:,:,c][:, :, None] * Vr[:,:,c][None, :, :] * Wr[:,:,c][None, None, :]
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err <= tol, max_err

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_random_search(n, m, p, max_rank, time_budget, seed=42):
    """Try to find a decomposition with rank <= max_rank via random initialization + local search."""
    rng = np.random.RandomState(seed)
    best_result = None
    best_rank = max_rank + 1
    
    for trial in range(50):
        if _time_remaining() < 5:
            break
        
        # Try a random rank
        r = rng.randint(max(3, max_rank - 10), max_rank + 1)
        if r >= best_rank:
            continue
        
        # Initialize randomly
        U = rng.randn(n*m, r) * 0.1
        V = rng.randn(m*p, r) * 0.1
        W = rng.randn(p*n, r) * 0.1
        
        # Simple gradient-free local search: try small perturbations
        # This is very slow for large dimensions, so limit iterations
        best_err = float('inf')
        for it in range(200):
            if _time_remaining() < 5:
                break
            valid, err = _verify_fast(U, V, W, n, m, p, tol=1e-4)
            if valid:
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
            if err < best_err:
                best_err = err
            # Random perturbation
            idx = rng.randint(0, n*m)
            c = rng.randint(0, r)
            U[idx, c] += rng.randn() * 0.05
        # End local search
    
    return best_result, best_rank

def _try_structural_search(n, m, p, time_budget):
    """
    Try structural decompositions. For 3x3 * 3x8, the tensor has a specific structure.
    We can try to exploit the fact that p=8 is larger.
    
    Key insight: The multiplication tensor T_{i,j,k} for C[i,k] = sum_j A[i,j]*B[j,k]
    has T[i,j,k] = 1 for all (i,j,k) in [0,n)x[0,m)x[0,p).
    
    We look for rank-r decompositions where r < n*m*p = 72.
    """
    n, m, p = 3, 3, 8
    best_result = None
    best_rank = n*m*p  # schoolbook
    
    # Try ranks from a lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # Actually the tensor rank lower bound is max(n*m, m*p, n*p) = 24
    # But for this specific tensor, we can do better.
    
    # Strategy: Use the fact that for fixed j, the sub-tensor over (i,k) is outer product of
    # vectors. T[i,j,k] = 1 means for each j, the (n x p) slice is all-ones.
    
    # A rank-r decomposition: sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1 for all i,j,k
    
    # Let's try a structured approach:
    # For each j in [0,m), we need sum_c u_c[i,j]*v_c[j,k]*w_c[k,i] = 1 for all i,k.
    
    # Simple approach: try rank = m * (something small)
    # For m=3, if we can do it with 3 "groups" each handling one j,
    # each group needs rank r_j such that sum_{c in group_j} u_c[i,j]*v_c[j,k]*w_c[k,i] = 1
    
    # For a single j: sum_c u_c[i]*v_c[k]*w_c[k,i] = 1 for all i in [0,3), k in [0,8)
    # This is like decomposing the all-ones 3x8 matrix as a sum of rank-1 tensors in a specific way.
    
    # Actually, let me think differently. Let's use numpy to solve small linear systems.
    
    # For rank r, we have r^3 unknowns (entries of U, V, W) and n*m*p = 72 equations.
    # For r >= 5, r^3 >= 125 > 72, so solutions may exist.
    
    # Let's try a systematic search for small ranks using scipy optimization.
    
    from scipy.optimize import minimize
    
    for target_rank in range(24, best_rank):
        if _time_remaining() < 10:
            break
        if target_rank >= best_rank:
            break
        
        # Parameterize: flatten U, V, W
        # U: 9 x r, V: 24 x r, W: 24 x r
        total_params = (n*m + m*p + p*n) * target_rank
        
        def objective(params):
            if _time_remaining() < 5:
                return 1e10
            U = params[:n*m*target_rank].reshape(n*m, target_rank)
            V = params[n*m*target_rank:(n*m+m*p)*target_rank].reshape(m*p, target_rank)
            W = params[(n*m+m*p)*target_rank:].reshape(p*n, target_rank)
            
            # Compute error
            Ur = U.reshape(n, m, target_rank)
            Vr = V.reshape(m, p, target_rank)
            Wr = W.reshape(p, n, target_rank)
            
            T = np.zeros((n, m, p))
            for c in range(target_rank):
                T += Ur[:,:,c][:, :, None] * Vr[:,:,c][None, :, :] * Wr[:,:,c][None, None, :]
            
            err = np.sum((T - 1.0)**2)
            return err
        
        # Initial guess: use a structured initialization
        # For each j, assign a subset of columns
        cols_per_j = target_rank // m
        params = np.zeros(total_params)
        
        # Initialize with schoolbook-like structure
        for j in range(m):
            for c in range(cols_per_j):
                global_c = j * cols_per_j + c
                # U[i*m+j, global_c] = 1 for some i
                # V[j*p+k, global_c] = 1 for some k  
                # W[k*n+i, global_c] = 1
                # This won't work directly; use random init instead
        
        params = np.random.randn(total_params) * 0.1
        
        # Try a few random restarts
        found = False
        for restart in range(3):
            if _time_remaining() < 10:
                break
            
            x0 = np.random.randn(total_params) * 0.1
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < 1e-6:
                    U = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V = res.x[n*m*target_rank:(n*m+m*p)*target_rank].reshape(m*p, target_rank)
                    W = res.x[(n*m+m*p)*target_rank:].reshape(p*n, target_rank)
                    
                    # Verify
                    valid, err = _verify_fast(U, V, W, n, m, p, tol=1e-5)
                    if valid:
                        best_rank = target_rank
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
    
    return best_result, best_rank

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    fallback = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = fallback
    
    # Try structural search
    try:
        if _time_remaining() > 20:
            result, rank = _try_structural_search(n, m, p, time_budget=_time_remaining() - 10)
            if result is not None and rank < best_rank:
                best_rank = rank
                best_result = result
    except Exception:
        pass
    
    # If still at schoolbook, try random search with limited budget
    if best_rank == n * m * p and _time_remaining() > 15:
        try:
            result, rank = _try_random_search(n, m, p, max_rank=n*m*p - 5, 
                                               time_budget=_time_remaining() - 10, seed=123)
            if result is not None and rank < best_rank:
                best_rank = rank
                best_result = result
        except Exception:
            pass
    
    # Final safety: ensure we return a valid result
    if _time_remaining() < 5:
        return fallback
    
    # Verify the best result
    try:
        valid, err = _verify_fast(best_result["U"], best_result["V"], best_result["W"], n, m, p, tol=1e-5)
        if not valid:
            return fallback
    except Exception:
        return fallback
    
    return best_result

# Ensure we don't exceed time on import
if __name__ == "__main__":
    pass
