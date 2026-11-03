"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomp(U, V, W, tol=1e-6):
    """Verify that the decomposition is exact for the tensor defined by the seed."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    # Check: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _compute_loss_and_grad(params, n, m, p, rank):
    """Compute loss and gradients for flattened parameters."""
    # Unpack
    U_size = n * m * rank
    V_size = m * p * rank
    W_size = p * n * rank
    
    U = params[:U_size].reshape(n * m, rank)
    V = params[U_size:U_size+V_size].reshape(m * p, rank)
    W = params[U_size+V_size:].reshape(p * n, rank)
    
    loss = 0.0
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    # Vectorized gradient computation is hard due to the 3-way product structure
    # We use a loop over i,j,k but vectorize over r
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                # Product vector: U[idx_u, :] * V[idx_v, :] * W[idx_w, :]
                prod = U[idx_u, :] * V[idx_v, :] * W[idx_w, :]
                val = np.sum(prod)
                err = val - 1.0
                loss += err * err
                
                # Gradients
                # dU[idx_u, r] = 2 * err * V[idx_v, r] * W[idx_w, r]
                # dV[idx_v, r] = 2 * err * U[idx_u, r] * W[idx_w, r]
                # dW[idx_w, r] = 2 * err * U[idx_u, r] * V[idx_v, r]
                
                dU[idx_u, :] += 2 * err * V[idx_v, :] * W[idx_w, :]
                dV[idx_v, :] += 2 * err * U[idx_u, :] * W[idx_w, :]
                dW[idx_w, :] += 2 * err * U[idx_u, :] * V[idx_v, :]
    
    grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    return loss, grad

def _try_optimization(target_rank, max_iter=2000, deadline=140.0, start_time=0):
    n, m, p = DIMENSIONS
    
    # Initialize
    np.random.seed(42 + target_rank)
    
    # Small random initialization
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Use L-BFGS-B
    res = minimize(_compute_loss_and_grad, params, args=(n, m, p, target_rank), method='L-BFGS-B', jac=True, options={'maxiter': max_iter, 'maxfun': 10000})
    
    if res.success or res.fun < 1e-10:
        final_params = res.x
        U_final = final_params[:n*m*target_rank].reshape(n*m, target_rank)
        V_final = final_params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_final = final_params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
        
        if _verify_decomp(U_final, V_final, W_final):
            return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
            
    return None

def construct():
    start_time = time.time()
    deadline = 145.0
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p # 75
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try ranks from 70 down to 60
    # If we find a lower rank, we can stop or continue
    for target_rank in range(70, 59, -1):
        if time.time() > deadline - 10:
            break
            
        result = _try_optimization(target_rank, max_iter=1000, deadline=deadline - 5, start_time=start_time)
        
        if result is not None:
            best_result = result
            best_rank = target_rank
            # Found a valid one, try to go lower
            continue
    
    if best_result is not None:
        return best_result
        
    return _schoolbook()
# EVOLVE-BLOCK-END
