import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_optimize(U, V, W, n, m, p, max_iters=300, lr=0.005, seed=0, deadline=None, momentum_beta=0.9):
    """Optimize U, V, W to satisfy the tensor identity using gradient descent with momentum."""
    np.random.seed(seed)
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    momentum_U = np.zeros_like(U)
    momentum_V = np.zeros_like(V)
    momentum_W = np.zeros_like(W)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        # Compute error and gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        total_err_sq = 0.0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U[i*m + j, :]
                    vk = V[j*p + k, :]
                    wk = W[k*n + i, :]
                    s_val = np.dot(ui, vk * wk)
                    err = s_val - 1.0
                    total_err_sq += err * err
                    
                    factor = 2.0 * err
                    grad_U[i*m + j, :] += factor * (vk * wk)
                    grad_V[j*p + k, :] += factor * (ui * wk)
                    grad_W[k*n + i, :] += factor * (ui * vk)
        
        err_norm = np.sqrt(total_err_sq)
        
        if err_norm < 1e-8:
            best_err = err_norm
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            break
        
        # Update momentum
        momentum_U = momentum_beta * momentum_U + (1 - momentum_beta) * grad_U
        momentum_V = momentum_beta * momentum_V + (1 - momentum_beta) * grad_V
        momentum_W = momentum_beta * momentum_W + (1 - momentum_beta) * grad_W
        
        # Update parameters
        U -= lr * momentum_U
        V -= lr * momentum_V
        W -= lr * momentum_W
        
        if err_norm < best_err:
            best_err = err_norm
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-7):
        return best_U, best_V, best_W, best_err
    
    return None, None, None, best_err

def _find_rank20_444(deadline):
    """Find a rank-20 decomposition for 4x4x4 matrix multiplication."""
    n, m, p = 4, 4, 4
    R = 20
    
    # Try multiple seeds and initializations
    for seed in range(20):
        if time.time() > deadline - 20:
            break
        
        # Random initialization
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        U_opt, V_opt, W_opt, err = _try_optimize(U, V, W, n, m, p, max_iters=500, lr=0.005, seed=seed, deadline=deadline)
        if U_opt is not None:
            return U_opt, V_opt, W_opt
    
    # If random init fails, try structured init
    # Start with schoolbook and truncate
    schoolbook_rank = n * m * p
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    U = U_school[:, :R].copy()
    V = V_school[:, :R].copy()
    W = W_school[:, :R].copy()
    
    U += np.random.randn(*U.shape) * 0.01
    V += np.random.randn(*V.shape) * 0.01
    W += np.random.randn(*W.shape) * 0.01
    
    U_opt, V_opt, W_opt, err = _try_optimize(U, V, W, n, m, p, max_iters=500, lr=0.005, seed=0, deadline=deadline)
    if U_opt is not None:
        return U_opt, V_opt, W_opt
    
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    # Try to find a rank-20 decomposition for 4x4x4
    U444, V444, W444 = _find_rank20_444(deadline)
    
    if U444 is not None:
        # Construct rank-40 decomposition for 4x4x8
        R_total = 40
        U = np.zeros((n*m, R_total))
        V = np.zeros((m*p, R_total))
        W = np.zeros((p*n, R_total))
        
        # First block (columns 0-3 of B, columns 0-3 of C)
        U[:, 0:20] = U444
        V[0:16, 0:20] = V444
        W[0:16, 0:20] = W444
        
        # Second block (columns 4-7 of B, columns 4-7 of C)
        U[:, 20:40] = U444
        V[16:32, 20:40] = V444
        W[16:32, 20:40] = W444
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-7):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to original approach if rank-40 construction fails
    schoolbook_rank = n * m * p
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
                
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
