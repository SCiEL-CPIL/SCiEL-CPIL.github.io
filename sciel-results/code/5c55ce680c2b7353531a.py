"""Matrix multiplication 3x5x5: rank-minimal exact tensor decomposition."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify exact decomposition."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    r = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != r:
            return False
    for row in V:
        if len(row) != r:
            return False
    for row in W:
        if len(row) != r:
            return False
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check non-zero positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    
    # Check zero positions (to ensure exact tensor decomposition)
    # Create a set of valid (i,j,k) triples mapped to (alpha, beta, gamma)
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_set.add((a, b, c))
    
    # Iterate over all possible indices
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in valid_set:
                    s = np.dot(U_np[a], V_np[b] * W_np[c])
                    if not np.isclose(s, 0.0, atol=1e-9):
                        return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _prepare_indices(n, m, p):
    """Precompute valid and invalid indices for loss calculation."""
    valid_idx = []
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_idx.append((a, b, c))
                valid_set.add((a, b, c))
    
    invalid_idx = []
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in valid_set:
                    invalid_idx.append((a, b, c))
    return valid_idx, invalid_idx

def _loss_and_grad(params, n, m, p, r, valid_idx, invalid_idx, lambda_zero=1.0):
    """
    Compute loss and gradient for the optimization.
    params: flattened U, V, W
    """
    U = params[:n*m*r].reshape(n*m, r)
    V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
    W = params[n*m*r + m*p*r : ].reshape(p*n, r)
    
    # Compute S = sum_r U_ar V_br W_cr
    # Using einsum for speed
    S = np.einsum('ar,br,cr->abc', U, V, W)
    
    loss = 0.0
    # Gradient accumulation
    gU = np.zeros_like(U)
    gV = np.zeros_like(V)
    gW = np.zeros_like(W)
    
    # Valid indices: target 1
    for a, b, c in valid_idx:
        s_val = S[a, b, c]
        err = s_val - 1.0
        loss += err * err
        
        # Gradient: dL/dU_ar = 2 * err * V_br * W_cr
        # We can vectorize this, but loop is okay for small counts (75)
        gU[a] += 2.0 * err * (V[b] * W[c])
        gV[b] += 2.0 * err * (U[a] * W[c])
        gW[c] += 2.0 * err * (U[a] * V[b])
        
    # Invalid indices: target 0
    for a, b, c in invalid_idx:
        s_val = S[a, b, c]
        loss += lambda_zero * s_val * s_val
        
        gU[a] += 2.0 * lambda_zero * s_val * (V[b] * W[c])
        gV[b] += 2.0 * lambda_zero * s_val * (U[a] * W[c])
        gW[c] += 2.0 * lambda_zero * s_val * (U[a] * V[b])
        
    grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
    return loss, grad

def _try_rank(target_rank, n, m, p, timeout=20, max_iter=100):
    """
    Try to find a decomposition of target_rank.
    """
    start_time = time.time()
    valid_idx, invalid_idx = _prepare_indices(n, m, p)
    
    for trial in range(5): # 5 restarts
        if time.time() - start_time > timeout:
            break
            
        # Initialize with random values
        np.random.seed(42 + trial)
        U_init = np.random.randn(n * m, target_rank) * 0.5
        V_init = np.random.randn(m * p, target_rank) * 0.5
        W_init = np.random.randn(p * n, target_rank) * 0.5
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Optimization
        try:
            res = minimize(
                _loss_and_grad, 
                params_init, 
                args=(n, m, p, target_rank, valid_idx, invalid_idx, 1.0),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            continue
            
        if time.time() - start_time > timeout:
            break
            
        if res.success or res.fun < 1e-6:
            # Extract optimized U, V, W
            U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_opt = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_opt = res.x[n*m*target_rank + m*p*target_rank : ].reshape(p*n, target_rank)
            
            # Snap to nearest half-integer
            U_snap = np.round(U_opt * 2) / 2
            V_snap = np.round(V_opt * 2) / 2
            W_snap = np.round(W_opt * 2) / 2
            
            # Verify
            if _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    return None

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    # 1. Schoolbook baseline
    best = _schoolbook()
    best_rank = n * m * p
    
    # 2. Try lower ranks
    # Start from 70 down to 30
    for target_rank in range(70, 25, -1):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time() - 10
        timeout = min(remaining / 5, 15) # Allow up to 15s per rank
        
        res = _try_rank(target_rank, n, m, p, timeout=timeout)
        if res:
            return res
            
    return best
# EVOLVE-BLOCK-END
