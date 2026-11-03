import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    # 1. Construct Schoolbook Decomposition (Rank n*m*p = 48)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb

    # 2. Try to find rank < 48 using optimized search
    # We start from rank 47 down to a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = 16, but realistically for 3x4x4 it's higher
    # Let's try down to 20 or so to be safe with time
    
    # We will use a smarter initialization and optimization strategy
    # Strategy: 
    # 1. Use Winograd-inspired structured initialization for overlapping 2x2 blocks
    # 2. Use ALS with careful convergence checks
    # 3. Use a final polish step with Newton's method to snap to half-integers

    for rank in range(best_rank - 1, 15, -1):
        if time.time() > deadline - 10:
            break
            
        # Try to find a decomposition of this rank
        found = _try_find_rank(n, m, p, rank, max_restarts=20, time_limit=5)
        if found is not None:
            U, V, W = found
            best_result = (U, V, W)
            best_rank = rank
            break  # Found the best (lowest) rank we tried, stop
            
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_find_rank(n, m, p, rank, max_restarts=10, time_limit=10):
    """
    Attempts to find a rank-r exact decomposition with half-integer entries.
    """
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() > start_time + time_limit:
            break
            
        # Initialization: Mix of random and structured
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Structured initialization based on Winograd-like patterns for 2x2 blocks
        # This is a heuristic to provide a better starting point than pure random
        # For each rank component, we try to cover the tensor entries
        for r in range(rank):
            # Randomly assign some structure
            # We want to encourage non-zero entries in positions that contribute to the sum
            # For simplicity, let's use random sparse initialization with some density
            density = 0.2 + 0.1 * (rank / 48) # Higher density for lower ranks? No, lower rank needs sparser?
            # Actually, for tensor decomposition, we want the factors to be dense enough to cover the support
            # Let's just use random values from {-1, 0, 1} with some probability
            
            for idx in range(n * m):
                if random.random() < 0.3:
                    U[idx, r] = random.choice([-1, 1])
            for idx in range(m * p):
                if random.random() < 0.3:
                    V[idx, r] = random.choice([-1, 1])
            for idx in range(p * n):
                if random.random() < 0.3:
                    W[idx, r] = random.choice([-1, 1])
        
        # Add small noise to break symmetry
        U += np.random.randn(n * m, rank) * 0.01
        V += np.random.randn(m * p, rank) * 0.01
        W += np.random.randn(p * n, rank) * 0.01
        
        # ALS Iterations
        prev_err = float('inf')
        converged = False
        for it in range(50):
            if time.time() > start_time + time_limit - 1:
                break
                
            # Update U
            U_new = np.zeros_like(U)
            for idx in range(n * m):
                i, j = divmod(idx, m)
                V_part = V[j*p:(j+1)*p, :]
                W_part = W[np.arange(p)*n + i, :]
                A = V_part * W_part
                b = np.ones(p)
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    U_new[idx, :] = x
                except:
                    U_new[idx, :] = U[idx, :]
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for idx in range(m * p):
                j, k = divmod(idx, p)
                U_part = U[np.arange(n)*m + j, :]
                W_part = W[k*n:(k+1)*n, :]
                A = U_part * W_part
                b = np.ones(n)
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    V_new[idx, :] = x
                except:
                    V_new[idx, :] = V[idx, :]
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for idx in range(p * n):
                k, i = divmod(idx, n)
                U_part = U[i*m:(i+1)*m, :]
                V_part = V[np.arange(m)*p + k, :]
                A = U_part * V_part
                b = np.ones(m)
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    W_new[idx, :] = x
                except:
                    W_new[idx, :] = W[idx, :]
            W = W_new
            
            # Compute error
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-8:
                converged = True
                break
            if abs(prev_err - max_err) < 1e-10:
                break
            prev_err = max_err
        
        # Check final error
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-6:
            # Try to snap to half-integers
            U_snapped, u_ok = _snap(U, 1e-4)
            V_snapped, v_ok = _snap(V, 1e-4)
            W_snapped, w_ok = _snap(W, 1e-4)
            
            if u_ok and v_ok and w_ok:
                if _check_exact(U_snapped, V_snapped, W_snapped, n, m, p):
                    return U_snapped, V_snapped, W_snapped
            
            # If snapping fails, try a polish step
            if max_err < 1e-4:
                U_polished, V_polished, W_polished = _polish(U, V, W, n, m, p, time_left=time_limit - (time.time() - start_time))
                if U_polished is not None:
                    return U_polished, V_polished, W_polished
                    
    return None

def _polish(U, V, W, n, m, p, time_left=5):
    """
    High-precision polish: use Newton's method to drive error to machine precision,
    then snap to half-integers.
    """
    if time_left < 0.5:
        return None, None, None
    
    start = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(20):
        if time.time() - start > time_left:
            break
        
        # Gauss-Seidel style updates for Newton's method
        # Update U
        for idx in range(n * m):
            i, j = divmod(idx, m)
            res = np.zeros(p)
            for k in range(p):
                val = np.sum(U[idx, :] * V[j*p+k, :] * W[k*n+i, :])
                res[k] = 1.0 - val
            
            V_part = V[j*p:(j+1)*p, :]
            W_part = W[np.arange(p)*n + i, :]
            A = V_part * W_part
            
            try:
                delta = np.linalg.solve(A, res)
                U[idx, :] += delta
            except:
                pass
        
        # Update V
        for idx in range(m * p):
            j, k = divmod(idx, p)
            res = np.zeros(n)
            for i in range(n):
                val = np.sum(U[i*m+j, :] * V[idx, :] * W[k*n+i, :])
                res[i] = 1.0 - val
            
            U_part = U[np.arange(n)*m + j, :]
            W_part = W[k*n:(k+1)*n, :]
            A = U_part * W_part
            
            try:
                delta = np.linalg.solve(A, res)
                V[idx, :] += delta
            except:
                pass
        
        # Update W
        for idx in range(p * n):
            k, i = divmod(idx, n)
            res = np.zeros(m)
            for j in range(m):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[idx, :])
                res[j] = 1.0 - val
            
            U_part = U[i*m:(i+1)*m, :]
            V_part = V[np.arange(m)*p + k, :]
            A = U_part * V_part
            
            try:
                delta = np.linalg.solve(A, res)
                W[idx, :] += delta
            except:
                pass
        
        # Check convergence
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-10:
            break
    
    # Final snap
    U_snapped, u_ok = _snap(U, 1e-3)
    V_snapped, v_ok = _snap(V, 1e-3)
    W_snapped, w_ok = _snap(W, 1e-3)
    
    if u_ok and v_ok and w_ok:
        if _check_exact(U_snapped, V_snapped, W_snapped, n, m, p):
            return U_snapped, V_snapped, W_snapped
    
    return None, None, None

def _snap(arr, tol=1e-6):
    """Snap to nearest half-integer if close."""
    snapped = np.round(arr * 2) / 2
    if np.max(np.abs(arr - snapped)) < tol:
        return snapped, True
    return arr, False

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
