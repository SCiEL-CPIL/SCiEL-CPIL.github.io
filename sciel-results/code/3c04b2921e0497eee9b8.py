import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

def check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Checks if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    R = U.shape[1]
    U_3d = U.reshape(n, m, R)
    V_3d = V.reshape(m, p, R)
    W_3d = W.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    if not np.allclose(T, np.ones((n, m, p)), atol=tol):
        return False
    return True

def compute_objective(U, V, W, n, m, p):
    """Compute the sum of squares of the residual."""
    R = U.shape[1]
    U_3d = U.reshape(n, m, R)
    V_3d = V.reshape(m, p, R)
    W_3d = W.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    residual = T - np.ones((n, m, p))
    return np.sum(residual ** 2)

def gradient_step(U, V, W, n, m, p, lr=0.01):
    """
    Perform one gradient descent step on the rank-R decomposition.
    """
    R = U.shape[1]
    U_3d = U.reshape(n, m, R)
    V_3d = V.reshape(m, p, R)
    W_3d = W.reshape(p, n, R)
    
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    residual = T - np.ones((n, m, p))
    
    # Gradients
    dU = np.einsum('jkr,kir,ijk->ijr', V_3d, W_3d, residual)
    dV = np.einsum('ijr,kir,ijk->jkr', U_3d, W_3d, residual)
    dW = np.einsum('ijr,jkr,ijk->kir', U_3d, V_3d, residual)
    
    U_new = U - lr * dU.reshape(n * m, R)
    V_new = V - lr * dV.reshape(m * p, R)
    W_new = W - lr * dW.reshape(p * n, R)
    
    return U_new, V_new, W_new

def try_rank(R_target, n, m, p, max_iters=5000, tol=1e-8, time_limit=10, n_restarts=3):
    """
    Try to find a rank-R_target decomposition using gradient descent.
    Returns (success, U, V, W) if successful.
    """
    start_time = time.time()
    
    for restart in range(n_restarts):
        # Initialize with random values
        np.random.seed(restart * 1000 + R_target + 42)
        U = np.random.randn(n * m, R_target) * 0.1
        V = np.random.randn(m * p, R_target) * 0.1
        W = np.random.randn(p * n, R_target) * 0.1
        
        best_obj = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        lr = 0.003
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            U, V, W = gradient_step(U, V, W, n, m, p, lr=lr)
            
            obj = compute_objective(U, V, W, n, m, p)
            
            if obj < best_obj:
                best_obj = obj
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            # Adaptive learning rate
            if obj > best_obj * 1.05:
                lr *= 0.7
            elif obj < best_obj * 0.95:
                lr *= 1.05
                lr = min(lr, 0.005)
            
            if obj < 1e-6:
                # Try polishing
                U, V, W = best_U, best_V, best_W
                for polish_it in range(200):
                    U, V, W = gradient_step(U, V, W, n, m, p, lr=0.0005)
                    obj = compute_objective(U, V, W, n, m, p)
                    if obj < 1e-10:
                        if check_decomposition(U, V, W, n, m, p, tol=1e-9):
                            return True, U, V, W
                        break
                break
        
        # Check best solution from this restart
        if best_obj < 1e-5:
            U, V, W = best_U, best_V, best_W
            if check_decomposition(U, V, W, n, m, p, tol=1e-7):
                # Try to polish
                for polish_it in range(300):
                    U, V, W = gradient_step(U, V, W, n, m, p, lr=0.0003)
                    obj = compute_objective(U, V, W, n, m, p)
                    if obj < 1e-10:
                        if check_decomposition(U, V, W, n, m, p, tol=1e-9):
                            return True, U, V, W
                        break
    
    return False, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Try ranks from 40 down to 30
    for R_target in [40, 38, 36, 35, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 15:
            break
        
        success, U, V, W = try_rank(R_target, n, m, p, max_iters=3000, tol=1e-8, time_limit=12, n_restarts=3)
        
        if success:
            # Verify with tighter tolerance
            if check_decomposition(U, V, W, n, m, p, tol=1e-9):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = R_target
                # Continue trying to go lower
            else:
                pass
    
    return best_result
