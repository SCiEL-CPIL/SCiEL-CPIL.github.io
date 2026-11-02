import numpy as np
import time

def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0
                    for r in range(rank):
                        s += W[k*n+i][r] * U[i*m+j1][r] * V[j2*p+k][r]
                    target = 1 if j1 == j2 else 0
                    if abs(s - target) > 1e-10:
                        return False
    return True

def _try_rank(rank, n, m, p, deadline, max_iters=5000, n_restarts=20):
    """Try to find a rank-rank decomposition using gradient descent."""
    np.random.seed(42)
    best_loss = float('inf')
    best_solution = None
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        # Try to optimize
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute loss and gradients
            loss = 0
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            target = 1.0 if j1 == j2 else 0.0
                            s = 0
                            ds_dU = 0
                            ds_dV = 0
                            ds_dW = 0
                            for r in range(rank):
                                term = W[k*n+i][r] * U[i*m+j1][r] * V[j2*p+k][r]
                                s += term
                                ds_dU = W[k*n+i][r] * V[j2*p+k][r]
                                ds_dV = W[k*n+i][r] * U[i*m+j1][r]
                                ds_dW = U[i*m+j1][r] * V[j2*p+k][r]
                            err = s - target
                            loss += err**2
                            dU[i*m+j1] += 2*err*ds_dU
                            dV[j2*p+k] += 2*err*ds_dV
                            dW[k*n+i] += 2*err*ds_dW
            
            if loss < best_loss:
                best_loss = loss
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if loss < 1e-12:
                break
            
            # Gradient step
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
    
    if best_loss < 1e-10:
        U, V, W = best_solution
        return U.tolist(), V.tolist(), W.tolist()
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # 130s deadline with 20s margin
    
    # Try ranks from 15 up to 18
    for rank in range(15, n*m*p + 1):
        if time.time() > deadline:
            break
        
        result = _try_rank(rank, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            # Verify with exact arithmetic
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook verification failed")
        
    return {"U": U, "V": V, "W": W}
