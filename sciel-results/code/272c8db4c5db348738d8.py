"""Tensor decomposition for 2x3x9 matrix multiplication."""
import time
import numpy as np
import random

DIMENSIONS = [2, 3, 9]

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _is_decomposition_valid(U, V, W, n, m, p, tol=1e-7):
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(R):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_random_decomposition(rank, n, m, p, max_iter=50, tol=1e-7, deadline=None):
    """Try to find a valid decomposition of given rank using random init + local search."""
    for attempt in range(max_iter):
        if deadline and time.time() > deadline:
            return None
        
        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize to help convergence
        for c in range(rank):
            norm_u = np.linalg.norm(U[:, c])
            norm_v = np.linalg.norm(V[:, c])
            norm_w = np.linalg.norm(W[:, c])
            if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                # Scale so that product of norms is ~1
                scale = (norm_u * norm_v * norm_w) ** (1.0/3.0)
                U[:, c] /= scale
                V[:, c] /= scale
                W[:, c] /= scale
        
        # Simple local search: try to reduce error by adjusting entries
        # Error for entry (i,j,k) is: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1
        # We can do coordinate descent on the factor entries
        
        best_error = float('inf')
        for iteration in range(100):
            if deadline and time.time() > deadline:
                return None
            
            # Compute current errors
            errors = {}
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        e = s - 1.0
                        errors[(i,j,k)] = e
                        max_err = max(max_err, abs(e))
            
            if max_err < best_error:
                best_error = max_err
            
            if max_err < tol:
                return U.copy(), V.copy(), W.copy()
            
            # Try to improve by adjusting one factor entry at a time
            # Pick the worst entry
            worst_key = max(errors.keys(), key=lambda x: abs(errors[x]))
            i, j, k = worst_key
            e = errors[(i,j,k)]
            
            # Adjust U[i*m+j, c] for each c
            improved = False
            for c in range(rank):
                v_val = V[j*p+k, c]
                w_val = W[k*n+i, c]
                if abs(v_val * w_val) < 1e-10:
                    continue
                # We want to change U[i*m+j, c] to reduce error
                # Current contribution: U[i*m+j,c] * v_val * w_val
                # Desired: reduce e, so we want to decrease e
                # New U = U - alpha * e * (v_val * w_val) / (v_val * w_val)^2 * (v_val * w_val)
                # Simpler: U_new = U - e * (v_val * w_val) / sum_c' (v_val'*w_val')^2
                # But this is complex. Use gradient step:
                # dU[i*m+j,c] = -e * v_val * w_val
                step = -0.1 * e * v_val * w_val
                old_val = U[i*m+j, c]
                U[i*m+j, c] = old_val + step
                improved = True
            
            if not improved:
                break
        
        # Check if we got close
        if best_error < tol:
            return U.copy(), V.copy(), W.copy()
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    n, m, p = DIMENSIONS
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = best_U.shape[1]
    
    # Try ranks from just below schoolbook down to a lower bound
    # Lower bound: at least n*m*p / min(n,m,p) is a rough bound, but let's try from 53 down
    # Actually, for matrix multiplication, the rank must be at least max(n*m, m*p, p*n) in some cases
    # But let's just try a range
    
    # First, try a few ranks near the schoolbook to see if we can shave off a bit
    for rank in range(best_rank - 1, max(20, best_rank - 15), -1):
        if time.time() > deadline:
            break
        
        result = _try_random_decomposition(rank, n, m, p, max_iter=5, tol=1e-7, deadline=deadline)
        if result is not None:
            U, V, W = result
            if _is_decomposition_valid(U, V, W, n, m, p, tol=1e-7):
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                break  # Found a valid one, stop
    
    # If we found a better one, try to go even lower
    if best_rank < n * m * p:
        for rank in range(best_rank - 1, max(15, best_rank - 10), -1):
            if time.time() > deadline:
                break
            
            result = _try_random_decomposition(rank, n, m, p, max_iter=3, tol=1e-7, deadline=deadline)
            if result is not None:
                U, V, W = result
                if _is_decomposition_valid(U, V, W, n, m, p, tol=1e-7):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
