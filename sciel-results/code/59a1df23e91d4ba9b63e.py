"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]
DEADLINE = 120.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0  # 5s safety margin

def _is_valid_entry(x):
    """Check if x is an integer or half-integer."""
    x2 = 2 * x
    return abs(x2 - round(x2)) < 1e-9

def _snap_to_half(x):
    """Snap x to nearest half-integer."""
    return round(2 * x) / 2.0

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    if Ua.shape != (n*m, rank) or Va.shape != (m*p, rank) or Wa.shape != (p*n, rank):
        return False
    
    # Check all entries are integers or half-integers
    for arr in [Ua, Va, Wa]:
        if not np.allclose(2*arr, np.round(2*arr), atol=1e-9):
            return False
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def _try_rank_altsolve(n, m, p, rank, max_iter=300, seed=None, tol=1e-10):
    """
    Try to find a valid decomposition of given rank using alternating linear solve.
    This is more robust than gradient descent for this type of problem.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Initialize with random values
    U = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)], dtype=float)
    V = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)], dtype=float)
    W = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)], dtype=float)
    
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if _remaining() < 2.0:
            break
        
        # Update U: for each (i,j), solve linear system
        for i in range(n):
            for j in range(m):
                # For each k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # Let M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # Then M @ U[i*m+j, :] = ones(p)
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p + k] * W[k*n + i]
                try:
                    U[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), solve linear system
        for j in range(m):
            for k in range(p):
                # For each i, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # Let M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                # Then M @ V[j*p+k, :] = ones(n)
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m + j] * W[k*n + i]
                try:
                    V[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i), solve linear system
        for k in range(p):
            for i in range(n):
                # For each j, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # Let M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                # Then M @ W[k*n+i, :] = ones(m)
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m + j] * V[j*p + k]
                try:
                    W[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Compute error
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < tol:
            break
    
    if best_error < tol:
        # Snap to half-integers
        Ua_snapped = np.vectorize(_snap_to_half)(best_U)
        Va_snapped = np.vectorize(_snap_to_half)(best_V)
        Wa_snapped = np.vectorize(_snap_to_half)(best_W)
        
        U_out = Ua_snapped.tolist()
        V_out = Va_snapped.tolist()
        W_out = Wa_snapped.tolist()
        
        if _verify_fast(U_out, V_out, W_out, n, m, p):
            return U_out, V_out, W_out
        
        # Also try without snapping
        U_out_raw = best_U.tolist()
        V_out_raw = best_V.tolist()
        W_out_raw = best_W.tolist()
        
        if _verify_fast(U_out_raw, V_out_raw, W_out, n, m, p):
            return U_out_raw, V_out_raw, W_out_raw
    
    return None

def _try_rank_gradient(n, m, p, rank, max_iter=2000, seed=None):
    """Try to find a valid decomposition of given rank using gradient descent."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    U = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)], dtype=float)
    V = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)], dtype=float)
    W = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)], dtype=float)
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    lr = 0.01
    for it in range(max_iter):
        if _remaining() < 2.0:
            break
        
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = T - 1.0
        
        grad_U = np.einsum('ijk,jkr,kir->ijr', err, V3, W3)
        grad_V = np.einsum('ijk,ijr,kir->jkr', err, U3, W3)
        grad_W = np.einsum('ijk,ijr,jkr->kir', err, U3, V3)
        
        U3 -= lr * grad_U
        V3 -= lr * grad_V
        W3 -= lr * grad_W
        
        if np.max(np.abs(err)) < 1e-10:
            break
        
        if it % 200 == 0:
            lr *= 0.5
    
    Ua = U3.reshape(n*m, rank)
    Va = V3.reshape(m*p, rank)
    Wa = W3.reshape(p*n, rank)
    
    Ua_snapped = np.vectorize(_snap_to_half)(Ua)
    Va_snapped = np.vectorize(_snap_to_half)(Va)
    Wa_snapped = np.vectorize(_snap_to_half)(Wa)
    
    U_out = Ua_snapped.tolist()
    V_out = Va_snapped.tolist()
    W_out = Wa_snapped.tolist()
    
    if _verify_fast(U_out, V_out, W_out, n, m, p):
        return U_out, V_out, W_out
    
    U_out_raw = Ua.tolist()
    V_out_raw = Va.tolist()
    W_out_raw = Wa.tolist()
    
    if _verify_fast(U_out_raw, V_out_raw, W_out_raw, n, m, p):
        return U_out_raw, V_out_raw, W_out_raw
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    best = schoolbook
    best_rank = n * m * p
    
    # Lower bound for rank: max(n*m, m*p, n*p) = max(15, 30, 18) = 30
    min_rank = max(n*m, m*p, n*p)
    
    for rank in range(min_rank, best_rank + 1):
        if _remaining() < 10.0:
            break
        
        found = False
        
        # Try alternating linear solve first (more robust)
        for attempt in range(5):
            if _remaining() < 5.0:
                break
            
            result = _try_rank_altsolve(n, m, p, rank, max_iter=200, seed=rank*100+attempt)
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    found = True
                    break
        
        # Try gradient descent if alternating solve didn't work
        if not found:
            for attempt in range(5):
                if _remaining() < 5.0:
                    break
                
                result = _try_rank_gradient(n, m, p, rank, max_iter=1000, seed=rank*100+100+attempt)
                if result is not None:
                    U, V, W = result
                    if _verify_fast(U, V, W, n, m, p):
                        best = {"U": U, "V": V, "W": W}
                        best_rank = rank
                        found = True
                        break
        
        if found:
            # Found a valid decomposition, can stop
            break
    
    # Final check: ensure the best result is valid
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        best = schoolbook
    
    return best
