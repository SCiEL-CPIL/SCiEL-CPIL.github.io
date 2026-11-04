import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p  # 45
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    # Schoolbook decomposition
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
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try ranks from low to high
    # Known lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # Try a range of ranks
    target_ranks = [20, 22, 24, 25, 27, 30, 33, 36, 39, 42]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10.0:
            break
        
        result = _attempt_decomposition(n, m, p, target_rank, deadline)
        if result is not None:
            best_rank = target_rank
            best_U, best_V, best_W = result
            break
    
    if best_rank < rank_sb:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def _compute_tensor_error(U, V, W, n, m, p):
    """Compute sum of squared errors for the tensor decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            u_vec = U[i * m + j, :]
            for k in range(p):
                v_vec = V[j * p + k, :]
                w_vec = W[k * n + i, :]
                s = np.dot(u_vec, v_vec * w_vec)
                err += (s - 1.0) ** 2
    return err

def _compute_gradients(U, V, W, n, m, p):
    """Compute gradients of the error with respect to U, V, W."""
    U_grad = np.zeros_like(U)
    V_grad = np.zeros_like(V)
    W_grad = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = U[i * m + j, :]
                v_vec = V[j * p + k, :]
                w_vec = W[k * n + i, :]
                s = np.dot(u_vec, v_vec * w_vec)
                residual = s - 1.0
                factor = 2.0 * residual
                
                U_grad[i * m + j, :] += factor * v_vec * w_vec
                V_grad[j * p + k, :] += factor * u_vec * w_vec
                W_grad[k * n + i, :] += factor * u_vec * v_vec
    
    return U_grad, V_grad, W_grad

def _attempt_decomposition(n, m, p, rank, deadline):
    """Try to find a rank-`rank` decomposition using multiple random restarts."""
    np.random.seed(42)
    
    max_restarts = 10
    max_iterations = 500
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random initialization
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Normalize columns
        for r in range(rank):
            nu = np.linalg.norm(U[:, r])
            nv = np.linalg.norm(V[:, r])
            nw = np.linalg.norm(W[:, r])
            if nu > 1e-10:
                U[:, r] /= nu
            if nv > 1e-10:
                V[:, r] /= nv
            if nw > 1e-10:
                W[:, r] /= nw
        
        # Gradient descent with adaptive learning rate
        lr = 0.01
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
            
            err = _compute_tensor_error(U, V, W, n, m, p)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                break
            
            U_grad, V_grad, W_grad = _compute_gradients(U, V, W, n, m, p)
            
            # Adaptive learning rate: reduce if error increases
            U_new = U - lr * U_grad
            V_new = V - lr * V_grad
            W_new = W - lr * W_grad
            
            err_new = _compute_tensor_error(U_new, V_new, W_new, n, m, p)
            if err_new < err:
                U, V, W = U_new, V_new, W_new
                lr *= 1.1  # Increase learning rate slightly
            else:
                lr *= 0.5  # Decrease learning rate
        
        # Check if we found a good solution
        if best_err < 1e-8:
            # Snap to nearest half-integers
            U_snap = np.round(best_U * 2) / 2
            V_snap = np.round(best_V * 2) / 2
            W_snap = np.round(best_W * 2) / 2
            
            err_snap = _compute_tensor_error(U_snap, V_snap, W_snap, n, m, p)
            if err_snap < 1e-8:
                return U_snap, V_snap, W_snap
    
    return None

# Keep the original function names as required
def search_rank(n, m, p, target_rank, deadline):
    """Alias for _attempt_decomposition."""
    return _attempt_decomposition(n, m, p, target_rank, deadline)
