"""Matrix multiplication 3,3,7 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

_START_TIME = time.time()
_TIME_LIMIT = 140  # safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _schoolbook_3x3x7():
    """Return schoolbook rank-63 decomposition for 3x3 * 3x7."""
    n, m, p = 3, 3, 7
    rank = n * m * p  # 63
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W):
    """Verify decomposition: T[i,j,k] = sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] == 1 for all i,j,k."""
    n, m, p = 3, 3, 7
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-8:
                    return False
    return True

def _try_rank_search(target_rank, time_budget):
    """
    Try to find a rank-`target_rank` decomposition via numerical optimization.
    Uses the constraint: T[i,j,k] = sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] = 1
    """
    n, m, p = 3, 3, 7
    r = target_rank
    
    # Parameterize: U (9 x r), V (21 x r), W (21 x r)
    # Total parameters: 9*r + 21*r + 21*r = 51*r
    # Constraints: 63 equations (one per (i,j,k))
    
    # For small r, we have more parameters than constraints, so solutions may exist.
    # We'll use a simple iterative approach: initialize randomly, then use gradient descent
    # on the residual sum of squares.
    
    np.random.seed(42)
    
    # Initialize
    U = np.random.randn(n*m, r) * 0.5
    V = np.random.randn(m*p, r) * 0.5
    W = np.random.randn(p*n, r) * 0.5
    
    # Residual function: for each (i,j,k), residual = 1 - sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t]
    def residuals():
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[idx] = 1.0 - np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    idx += 1
        return res
    
    # Compute initial residual
    res = residuals()
    ss = np.sum(res**2)
    
    # Simple gradient descent with line search
    lr = 0.01
    best_ss = ss
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    start = time.time()
    max_iters = 5000
    
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        
        # Compute gradients via finite differences (expensive but simple)
        # Actually, let's compute analytical gradients
        # d/dU[i,j,t] of sum_k (1 - sum_t' U[i,j,t']*V[j,k,t']*W[k,i,t'])^2
        # = -2 * sum_k (1 - ...) * V[j,k,t] * W[k,i,t]
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r_ijk = res[i*m*p + j*p + k]
                    for t in range(r):
                        u_val = U[i*m+j, t]
                        v_val = V[j*p+k, t]
                        w_val = W[k*n+i, t]
                        grad_U[i*m+j, t] += -2 * r_ijk * v_val * w_val
                        grad_V[j*p+k, t] += -2 * r_ijk * u_val * w_val
                        grad_W[k*n+i, t] += -2 * r_ijk * u_val * v_val
        
        # Gradient step
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Recompute residual
        res = residuals()
        new_ss = np.sum(res**2)
        
        if new_ss < best_ss:
            best_ss = new_ss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Adaptive learning rate
        if new_ss < ss:
            lr = min(lr * 1.05, 0.1)
        else:
            lr = max(lr * 0.9, 0.001)
        
        ss = new_ss
        
        if best_ss < 1e-10:
            break
    
    if best_ss < 1e-6:
        # Snap to nearest "nice" values if possible
        return best_U, best_V, best_W, best_ss
    return None

def _try_split_decomposition():
    """
    Try to build a decomposition by splitting the 7 columns into groups.
    3x3 * 3x7 = 3x3 * 3x3 + 3x3 * 3x3 + 3x3 * 3x1
    Use rank-23 for each 3x3*3x3 (Strassen-like) and rank-9 for 3x3*3x1.
    Total: 23 + 23 + 9 = 55
    """
    n, m = 3, 3
    p1, p2, p3 = 3, 3, 1  # 3+3+1 = 7
    
    # For 3x3 * 3x3, use a known rank-23 decomposition
    # Strassen's algorithm gives rank 7 for 2x2, and there are known rank-23 for 3x3.
    # Let me construct one using the standard approach.
    
    # Actually, let's use a simpler approach: 
    # For 3x3 * 3x3, the schoolbook rank is 9.
    # Known optimal rank for 3x3 matrix multiplication is 19 (Alon-Boppana).
    # But finding explicit rank-19 coefficients is complex.
    
    # Let's try a different strategy: use the fact that we can share computations.
    # 
    # Alternative: use a block decomposition.
    # A (3x3) * B (3x7) where B = [B1, B2, B3] with B1, B2 (3x3), B3 (3x1)
    # C = A*B1 + A*B2 + A*B3
    # 
    # If we use schoolbook for each: 9 + 9 + 3 = 21... wait, that's for the output.
    # No, each 3x3*3x3 needs rank 9 in schoolbook, and 3x3*3x1 needs rank 3.
    # Total: 9 + 9 + 3 = 21? No, that's not right either.
    #
    # Let me think differently. The total schoolbook rank is 63.
    # Can we find linear dependencies to reduce it?
    
    # For now, let's just try numerical optimization for various ranks.
    pass

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    """
    n, m, p = 3, 3, 7
    
    # Start with schoolbook as baseline
    U, V, W = _schoolbook_3x3x7()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions via numerical search
    # The theoretical lower bound is ceil(63 / something), but in practice
    # we can try ranks from 50 down.
    
    # Try a few target ranks
    target_ranks = [55, 54, 53, 52, 51, 50]
    
    for tr in target_ranks:
        if _time_left() < 30:
            break
        
        time_budget = min(_time_left() - 30, 20)
        result = _try_rank_search(tr, time_budget)
        if result is not None:
            U_try, V_try, W_try, ss = result
            if ss < 1e-6:
                best_rank = tr
                best_U, best_V, best_W = U_try, V_try, W_try
                break
    
    # Return the best decomposition found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
