"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _compute_error_and_grad_vec(U, V, W, n, m, p):
    """Compute sum of squared errors and gradients using vectorized operations."""
    rank = U.shape[1]
    # Create all (i,j,k) combinations
    i_vals = np.repeat(np.arange(n), m * p)
    j_vals = np.tile(np.repeat(np.arange(m), p), n)
    k_vals = np.tile(np.arange(p), n * m)
    
    u_idx = i_vals * m + j_vals
    v_idx = j_vals * p + k_vals
    w_idx = k_vals * n + i_vals
    
    # Compute all partial products
    UVW = U[u_idx] * V[v_idx] * W[w_idx]
    
    # Sum over rank dimension to get tensor values
    tensor_vals = UVW.sum(axis=1)
    
    # Error
    errors = tensor_vals - 1.0
    err = float(np.sum(errors * errors))
    
    # Gradients
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for c in range(rank):
        grad_U[u_idx, c] += 2.0 * errors * V[v_idx, c] * W[w_idx, c]
        grad_V[v_idx, c] += 2.0 * errors * U[u_idx, c] * W[w_idx, c]
        grad_W[w_idx, c] += 2.0 * errors * U[u_idx, c] * V[v_idx, c]
    
    return err, grad_U, grad_V, grad_W

def _gradient_descent_vec(U, V, W, n, m, p, max_iters=500, lr=0.01, tol=1e-10, deadline=None):
    """Run vectorized gradient descent to minimize the error."""
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    beta = 0.9
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        err, gU, gV, gW = _compute_error_and_grad_vec(U, V, W, n, m, p)
        if err < tol:
            break
        
        vU = beta * vU + (1 - beta) * gU
        vV = beta * vV + (1 - beta) * gV
        vW = beta * vW + (1 - beta) * gW
        
        U -= lr * vU
        V -= lr * vV
        W -= lr * vW
    
    return U, V, W, err

def _try_rank(rank, n, m, p, deadline, seed=42):
    """Try to find a valid decomposition of given rank."""
    np.random.seed(seed)
    random.seed(seed)
    
    best_result = None
    best_err = float('inf')
    
    n_attempts = 5
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        scale = 0.5
        U = np.random.randn(n * m, rank) * scale
        V = np.random.randn(m * p, rank) * scale
        W = np.random.randn(p * n, rank) * scale
        
        for lr in [0.001, 0.005, 0.01, 0.02, 0.05]:
            if time.time() > deadline:
                break
            
            U_try, V_try, W_try, err = _gradient_descent_vec(
                U.copy(), V.copy(), W.copy(), n, m, p,
                max_iters=300, lr=lr, tol=1e-10, deadline=deadline
            )
            
            if err < best_err:
                best_err = err
                best_result = (U_try, V_try, W_try)
            
            if err < 1e-9:
                return U_try, V_try, W_try
    
    if best_result is not None and best_err < 1e-8:
        return best_result
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Known minimum rank for 3x3x4 is 19
    ranks_to_try = list(range(19, 26)) + list(range(26, best_rank))
    
    for r in ranks_to_try:
        if time.time() > deadline - 3:
            break
        
        result = _try_rank(r, n, m, p, deadline - 3, seed=r * 7 + 13)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify(U_r, V_r, W_r, n, m, p):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = r
                print(f"Found valid decomposition of rank {r}")
                if r == 19:
                    break
                continue
            else:
                print(f"Rank {r} result failed verification")
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
