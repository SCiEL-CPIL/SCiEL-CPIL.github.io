"""Matrix multiplication 3x5x7 - rank minimization via improved ALS and systematic local search."""
import time
import random
import numpy as np

def _verify_np(U, V, W, n, m, p):
    """Fast verification using numpy."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    try:
        result = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    except Exception:
        return False
        
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using fractions."""
    from fractions import Fraction
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = Fraction(1)
                val = Fraction(0)
                for c in range(rank):
                    u_val = Fraction(U[i*m+j][c])
                    v_val = Fraction(V[j*p+k][c])
                    w_val = Fraction(W[k*n+i][c])
                    val += u_val * v_val * w_val
                if val != target:
                    return False
    return True

def _snap_to_half_integer(arr):
    """Snap an array to the nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _calculate_error(U, V, W, n, m, p):
    """Calculate the sum of squared errors."""
    U_t = np.array(U, dtype=np.float64).reshape(n, m, -1)
    V_t = np.array(V, dtype=np.float64).reshape(m, p, -1)
    W_t = np.array(W, dtype=np.float64).reshape(p, n, -1)
    
    result = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    error = np.sum((result - 1.0)**2)
    return error

def _systematic_local_search(U, V, W, n, m, p, max_rounds=50):
    """
    Systematic local search: try changing each entry to its nearest half-integer neighbors.
    Returns updated U, V, W and final error.
    """
    nm, mp, pn = n*m, m*p, p*n
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    err = _calculate_error(U, V, W, n, m, p)
    
    for round_num in range(max_rounds):
        improved = False
        
        # Try U entries
        for i in range(nm):
            for c in range(U.shape[1]):
                current = U[i, c]
                # Neighbors: current-0.5, current+0.5
                for delta in [-0.5, 0.5]:
                    new_val = current + delta
                    U[i, c] = new_val
                    new_err = _calculate_error(U, V, W, n, m, p)
                    if new_err < err - 1e-12:
                        err = new_err
                        improved = True
                        break
                    else:
                        U[i, c] = current
        
        # Try V entries
        for i in range(mp):
            for c in range(V.shape[1]):
                current = V[i, c]
                for delta in [-0.5, 0.5]:
                    new_val = current + delta
                    V[i, c] = new_val
                    new_err = _calculate_error(U, V, W, n, m, p)
                    if new_err < err - 1e-12:
                        err = new_err
                        improved = True
                        break
                    else:
                        V[i, c] = current
        
        # Try W entries
        for i in range(pn):
            for c in range(W.shape[1]):
                current = W[i, c]
                for delta in [-0.5, 0.5]:
                    new_val = current + delta
                    W[i, c] = new_val
                    new_err = _calculate_error(U, V, W, n, m, p)
                    if new_err < err - 1e-12:
                        err = new_err
                        improved = True
                        break
                    else:
                        W[i, c] = current
        
        if not improved:
            break
    
    return U, V, W, err

def _als_decompose(r, n, m, p, max_iter=30, seed=42):
    """
    Perform ALS to find a real-valued rank-r decomposition.
    Returns U, V, W as numpy arrays.
    """
    np.random.seed(seed)
    nm, mp, pn = n*m, m*p, p*n
    
    U = np.random.randn(nm, r) * 0.1
    V = np.random.randn(mp, r) * 0.1
    W = np.random.randn(pn, r) * 0.1
    
    prev_error = float('inf')
    
    for it in range(max_iter):
        # Update U
        V_t = V.reshape(m, p, r)
        W_t = W.reshape(p, n, r)
        
        # B[i, j, k, c] = V[j, k, c] * W[k, i, c]
        B = np.einsum('jkc,kic->ijkc', V_t, W_t)
        
        U_new = np.zeros((n, m, r))
        for i in range(n):
            for j in range(m):
                try:
                    sol, _, _, _ = np.linalg.lstsq(B[i, j], np.ones(p), rcond=None)
                    U_new[i, j] = sol
                except:
                    U_new[i, j] = U[i*m+j, :]
        U = U_new.reshape(nm, r)
        
        # Update V
        U_t = U.reshape(n, m, r)
        W_t = W.reshape(p, n, r)
        
        # C[j, k, i, c] = U[i, j, c] * W[k, i, c]
        C = np.einsum('ijc,kic->jkic', U_t, W_t)
        
        V_new = np.zeros((m, p, r))
        for j in range(m):
            for k in range(p):
                try:
                    sol, _, _, _ = np.linalg.lstsq(C[j, k], np.ones(n), rcond=None)
                    V_new[j, k] = sol
                except:
                    V_new[j, k] = V[j*p+k, :]
        V = V_new.reshape(mp, r)
        
        # Update W
        U_t = U.reshape(n, m, r)
        V_t = V.reshape(m, p, r)
        
        # D[k, i, j, c] = U[i, j, c] * V[j, k, c]
        D = np.einsum('ijc,jkc->kijc', U_t, V_t)
        
        W_new = np.zeros((p, n, r))
        for k in range(p):
            for i in range(n):
                try:
                    sol, _, _, _ = np.linalg.lstsq(D[k, i], np.ones(m), rcond=None)
                    W_new[k, i] = sol
                except:
                    W_new[k, i] = W[k*n+i, :]
        W = W_new.reshape(pn, r)
        
        # Check convergence
        error = _calculate_error(U, V, W, n, m, p)
        if abs(prev_error - error) < 1e-10:
            break
        prev_error = error
        
    return U, V, W

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = 3, 5, 7
    nm, mp, pn = n*m, m*p, p*n  # 15, 35, 21
    
    # Schoolbook baseline: rank 105
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def try_rank(r, num_trials=10):
        """Try to find a decomposition of rank r using ALS + systematic local search."""
        nonlocal best_result, best_rank
        if r >= best_rank:
            return False
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return False
            
            seed = 1000 + r * 100 + trial
            U_real, V_real, W_real = _als_decompose(r, n, m, p, max_iter=25, seed=seed)
            
            # Snap to half-integers
            U_snap = _snap_to_half_integer(U_real)
            V_snap = _snap_to_half_integer(V_real)
            W_snap = _snap_to_half_integer(W_real)
            
            # Systematic local search
            U_ls, V_ls, W_ls, err = _systematic_local_search(U_snap, V_snap, W_snap, n, m, p, max_rounds=20)
            
            # Check if we found an exact solution
            if err < 1e-10:
                U_list = U_ls.tolist()
                V_list = V_ls.tolist()
                W_list = W_ls.tolist()
                
                if _verify_exact(U_list, V_list, W_list, n, m, p):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = r
                    return True
        
        return False
    
    # Try ranks from lower to higher
    # Start from a reasonable lower bound
    for r in range(40, 105):
        if time.time() > deadline:
            break
        if try_rank(r, num_trials=5):
            break
    
    return best_result
