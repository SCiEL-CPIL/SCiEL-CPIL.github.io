import time
import sys
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

def _schoolbook():
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
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

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 3, 3, 6
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    # Check if entries are half-integers
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _loss_and_grad(x, n, m, p, rank):
    """Calculate loss and gradient for the optimization problem."""
    U = x[0:n*m*rank].reshape(n*m, rank)
    V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
    W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
    
    loss = 0.0
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                residual = s - 1.0
                loss += residual * residual
                for r in range(rank):
                    grad_U[i*m + j, r] += 2 * residual * V[j*p + k, r] * W[k*n + i, r]
                    grad_V[j*p + k, r] += 2 * residual * U[i*m + j, r] * W[k*n + i, r]
                    grad_W[k*n + i, r] += 2 * residual * U[i*m + j, r] * V[j*p + k, r]
    
    grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    return loss, grad

def _try_rank(rank, time_limit=10.0, max_restarts=5):
    """Try to find a valid decomposition of the given rank."""
    n, m, p = 3, 3, 6
    total_vars = (n*m + m*p + p*n) * rank
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with small random integers or half-integers
        # Using a mix of 0, 0.5, -0.5, 1, -1
        rng = np.random.RandomState(restart + 12345)
        x0 = rng.choice([0, 0.5, -0.5, 1, -1], size=total_vars)
        
        # Try optimization
        try:
            res = minimize(_loss_and_grad, x0, args=(n, m, p, rank), method='L-BFGS-B', 
                           jac=True, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.success and res.fun < 1e-10:
                # Extract and round to nearest half-integer
                U_flat = res.x[0:n*m*rank].reshape(n*m, rank)
                V_flat = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_flat = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                U_rounded = np.round(U_flat * 2) / 2.0
                V_rounded = np.round(V_flat * 2) / 2.0
                W_rounded = np.round(W_flat * 2) / 2.0
                
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                if _verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
        except Exception:
            pass
            
    return None

def _try_rank_gradient(rank, time_limit=10.0, max_iters=100):
    """Try to find a valid decomposition using gradient descent with learning rate decay."""
    n, m, p = 3, 3, 6
    start_time = time.time()
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Initialize with random half-integers
    rng = np.random.RandomState(rank + 1000)
    U = rng.choice([0, 0.5, -0.5, 1, -1], size=(n*m, rank))
    V = rng.choice([0, 0.5, -0.5, 1, -1], size=(m*p, rank))
    W = rng.choice([0, 0.5, -0.5, 1, -1], size=(p*n, rank))
    
    lr = 0.05
    best_loss = float('inf')
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute residuals
        residual = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    residual[i, j, k] = s - 1.0
        
        loss = np.sum(residual ** 2)
        if loss < best_loss:
            best_loss = loss
            if loss < 1e-12:
                break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r_ijk = residual[i, j, k]
                    for r in range(rank):
                        dU[i * m + j, r] += 2 * r_ijk * V[j * p + k, r] * W[k * n + i, r]
                        dV[j * p + k, r] += 2 * r_ijk * U[i * m + j, r] * W[k * n + i, r]
                        dW[k * n + i, r] += 2 * r_ijk * U[i * m + j, r] * V[j * p + k, r]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Decay learning rate
        lr *= 0.98
    
    # Check if we found a valid solution
    if best_loss < 1e-9:
        # Convert to list of lists and round to half-integers
        U_rounded = np.round(U * 2) / 2.0
        V_rounded = np.round(V * 2) / 2.0
        W_rounded = np.round(W * 2) / 2.0
        
        U_list = U_rounded.tolist()
        V_list = V_rounded.tolist()
        W_list = W_rounded.tolist()
        
        if _verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # Leave 5s margin
    
    # Try to find lower ranks
    # Start from 53 and go down
    # Given the difficulty, we might only check a few
    for rank in range(53, 30, -1):
        if time.time() > deadline:
            break
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
            
        # Allocate time based on how many ranks are left
        # Simple allocation: fixed 5s per rank attempt
        result = _try_rank(rank, time_limit=min(5.0, remaining_time - 1.0), max_restarts=3)
        
        if result is not None:
            return result
        
        # Also try gradient descent approach
        if time.time() < deadline - 5:
            result = _try_rank_gradient(rank, time_limit=min(5.0, deadline - time.time() - 1.0), max_iters=50)
            if result is not None:
                return result
            
    # Fallback to schoolbook
    return _schoolbook()
