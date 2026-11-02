import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def _compute_tensor(U, V, W, n, m, p):
    """Compute the tensor from factor matrices."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
    return T

def _compute_gradients(U, V, W, n, m, p):
    """Compute gradients for U, V, W."""
    rank = U.shape[1]
    T = _compute_tensor(U, V, W, n, m, p)
    residual = T - np.ones((n, m, p))
    
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for c in range(rank):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = residual[i, j, k]
                    # grad_U[i*m+j, c]
                    grad_U[i*m+j, c] += 2 * r * v[j*p+k] * w[k*n+i]
                    # grad_V[j*p+k, c]
                    grad_V[j*p+k, c] += 2 * r * u[i*m+j] * w[k*n+i]
                    # grad_W[k*n+i, c]
                    grad_W[k*n+i, c] += 2 * r * u[i*m+j] * v[j*p+k]
    
    return grad_U, grad_V, grad_W

def _optimize_rank(n, m, p, rank, max_time=5.0, lr=0.001):
    """Optimize to find a decomposition of given rank."""
    start = time.time()
    np.random.seed(rank * 7919 + 42)
    
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    best_error = float('inf')
    best_state = (U.copy(), V.copy(), W.copy())
    
    for iteration in range(5000):
        if time.time() - start > max_time:
            break
        
        T = _compute_tensor(U, V, W, n, m, p)
        error = np.linalg.norm(T - np.ones((n, m, p)))
        
        if error < best_error:
            best_error = error
            best_state = (U.copy(), V.copy(), W.copy())
        
        if error < 1e-10:
            break
        
        grad_U, grad_V, grad_W = _compute_gradients(U, V, W, n, m, p)
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Adaptive learning rate
        if error > best_error * 1.1:
            lr *= 0.5
        elif error < best_error * 0.9:
            lr *= 1.1
    
    return best_state if best_error < 1e-8 else None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook
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
    
    best = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank
    for target_rank in range(35, 18, -1):
        if time.time() > deadline:
            break
        result = _optimize_rank(n, m, p, target_rank, max_time=4.0)
        if result:
            U, V, W = result
            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    
    return best
