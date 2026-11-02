import time
import numpy as np

DIMENSIONS = [2, 3, 4]
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return 150.0 - _DEADLINE_MARGIN - _elapsed()

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor T[i,j,k] = 1 if valid, else 0.
    Convention: C[i,k] = sum_j A[i,j] B[j,k]
    Tensor decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    So target T[i,j,k] = 1 for all i,j,k (since each (i,j,k) contributes to one product term)
    """
    T = np.ones((n, m, p))
    return T

def _tensor_error(U, V, W, n, m, p, T):
    """Compute the tensor decomposition error."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    rank = U.shape[1]
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j, :]
                v_val = V[j * p + k, :]
                w_val = W[k * n + i, :]
                # Sum over r of U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                term = np.sum(u_val * v_val * w_val)
                error += (term - T[i, j, k]) ** 2
    return error

def _try_rank(n, m, p, rank, max_time=30.0):
    """Try to find a rank-rank decomposition using random init + gradient descent."""
    T = _build_target_tensor(n, m, p)
    start = time.time()
    
    # Multiple random restarts
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    num_restarts = 5
    steps_per_restart = 500
    lr = 0.01
    
    for restart in range(num_restarts):
        if time.time() - start > max_time:
            break
        
        # Random initialization
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        for step in range(steps_per_restart):
            if time.time() - start > max_time:
                break
            
            # Compute error and gradients
            error = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        u_val = U[u_idx, :]
                        v_val = V[v_idx, :]
                        w_val = W[w_idx, :]
                        
                        term = np.sum(u_val * v_val * w_val)
                        err = term - T[i, j, k]
                        error += err ** 2
                        
                        # Gradient with respect to U[u_idx, r]
                        # d(err^2)/dU[u_idx,r] = 2*err * V[v_idx,r] * W[w_idx,r]
                        grad_U[u_idx, :] += 2 * err * v_val * w_val
                        grad_V[v_idx, :] += 2 * err * u_val * w_val
                        grad_W[w_idx, :] += 2 * err * u_val * v_val
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        if error < best_error:
            best_error = error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        # Check if we found an exact solution
        if best_error < 1e-10:
            break
    
    if best_error < 1e-10 and best_U is not None:
        return best_U, best_V, best_W, best_error
    return None, None, None, best_error

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24
    
    # Start with schoolbook decomposition as fallback
    U_fb = np.zeros((n * m, schoolbook_rank))
    V_fb = np.zeros((m * p, schoolbook_rank))
    W_fb = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
    
    # Try to find lower rank decompositions
    # Lower bounds: rank >= max(n*p, n*m, m*p) = max(8, 6, 12) = 12
    # Known: for 2x3x4, the minimal rank is likely 12 or slightly above
    
    best_rank = schoolbook_rank
    best_U = U_fb
    best_V = V_fb
    best_W = W_fb
    
    # Try ranks from lower bound up to schoolbook_rank - 1
    min_rank = max(n * p, n * m, m * p)  # 12
    
    for rank in range(min_rank, schoolbook_rank):
        if _time_left() < 10.0:
            break
        
        # Allocate time based on how many ranks we might try
        remaining_ranks = schoolbook_rank - rank
        time_per_rank = min(20.0, max(5.0, _time_left() / max(1, remaining_ranks)))
        
        U, V, W, error = _try_rank(n, m, p, rank, max_time=time_per_rank)
        
        if U is not None and error < 1e-10:
            best_rank = rank
            best_U = U
            best_V = V
            best_W = W
            break  # Found minimal rank, stop
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
