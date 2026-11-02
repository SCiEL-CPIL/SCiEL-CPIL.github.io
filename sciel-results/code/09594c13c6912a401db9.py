import time
import numpy as np

DIMENSIONS = [2, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 115.0  # safety margin before 120s kill

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor: T[i,j,k] = 1 for all valid (i,j,k)."""
    return np.ones((n, m, p))

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _compute_tensor(U, V, W, n, m, p):
    """Compute the tensor from U, V, W: T_est[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
    return T

def _loss(U, V, W, T, n, m, p):
    """Compute MSE loss."""
    T_est = _compute_tensor(U, V, W, n, m, p)
    return np.sum((T_est - T)**2)

def _coordinate_descent(U, V, W, T, n, m, p, lr=0.05, max_iters=2000):
    """Coordinate descent to minimize loss."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    r = U.shape[1]
    
    prev_loss = _loss(U, V, W, T, n, m, p)
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Update U
        for idx in range(n * m):
            i = idx // m
            j = idx % m
            for c in range(r):
                grad = 0.0
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    grad += (s - T[i,j,k]) * V[j*p+k, c] * W[k*n+i, c]
                U[idx, c] -= lr * grad
        
        # Update V
        for idx in range(m * p):
            j = idx // p
            k = idx % p
            for c in range(r):
                grad = 0.0
                for i in range(n):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    grad += (s - T[i,j,k]) * U[i*m+j, c] * W[k*n+i, c]
                V[idx, c] -= lr * grad
        
        # Update W
        for idx in range(p * n):
            k = idx // n
            i = idx % n
            for c in range(r):
                grad = 0.0
                for j in range(m):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    grad += (s - T[i,j,k]) * U[i*m+j, c] * V[j*p+k, c]
                W[idx, c] -= lr * grad
        
        if it % 100 == 0:
            curr_loss = _loss(U, V, W, T, n, m, p)
            if curr_loss < 1e-12:
                return U, V, W, curr_loss
            # Adaptive learning rate
            if curr_loss > prev_loss * 1.1:
                lr *= 0.8
            prev_loss = curr_loss
    
    final_loss = _loss(U, V, W, T, n, m, p)
    return U, V, W, final_loss

def _try_rank(n, m, p, r, T, num_restarts=5):
    """Try to find a rank-r decomposition."""
    best_U, best_V, best_W, best_loss = None, None, None, float('inf')
    
    for restart in range(num_restarts):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        np.random.seed(42 + restart * 1000)
        # Initialize with small random values
        U = np.random.randn(n * m, r) * 0.3
        V = np.random.randn(m * p, r) * 0.3
        W = np.random.randn(p * n, r) * 0.3
        
        # Run coordinate descent
        U, V, W, loss = _coordinate_descent(U, V, W, T, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-12:
            break
    
    if best_loss < 1e-10:
        # Verify
        if _verify_decomposition(best_U, best_V, best_W, n, m, p):
            return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    T = _build_tensor(n, m, p)
    
    # Try ranks from lower to higher, but start with some heuristics
    # For 2x3x6, try ranks that are plausible
    # Known: rank(2x2x2)=7, rank(2x3x3)=7
    # Try a range of ranks
    candidate_ranks = list(range(15, best_rank + 1))  # Try from 15 up to 36
    
    for r in candidate_ranks:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        result = _try_rank(n, m, p, r, T, num_restarts=3)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break  # Found the lowest rank in our search
    
    # Snap to half-integers if close
    def snap_half(x):
        x = np.array(x, dtype=float)
        snapped = np.round(x * 2) / 2.0
        return snapped
    
    U_final = snap_half(best_U)
    V_final = snap_half(best_V)
    W_final = snap_half(best_W)
    
    # Verify after snapping
    if _verify_decomposition(U_final, V_final, W_final, n, m, p):
        best_U, best_V, best_W = U_final, V_final, W_final
    else:
        # Try without snapping
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
