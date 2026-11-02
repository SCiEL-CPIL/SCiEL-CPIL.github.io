"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 112
    
    def tensor_residual(U, V, W):
        """Compute the residual of the tensor equation.
        T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        where T[i,j,k] = 1 for the multiplication tensor.
        """
        rank = U.shape[1]
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    residual += (s - 1.0) ** 2
        return residual
    
    def tensor_residual_vec(U, V, W):
        """Compute the residual vector for each (i,j,k)."""
        rank = U.shape[1]
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    res[i,j,k] = s - 1.0
        return res
    
    # Start with schoolbook
    rank = schoolbook_rank
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    best_residual = 0.0
    
    # Try to find lower rank decompositions using iterative optimization
    # Strategy: for each target rank, try random initializations and gradient descent
    
    target_ranks = list(range(80, schoolbook_rank + 1))
    random.seed(42)
    np.random.seed(42)
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        # Multiple random restarts for this rank
        max_restarts = 5
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U_try = np.random.randn(n*m, target_rank) * 0.1
            V_try = np.random.randn(m*p, target_rank) * 0.1
            W_try = np.random.randn(p*n, target_rank) * 0.1
            
            # Gradient descent with adaptive learning rate
            lr = 0.01
            best_local_residual = np.inf
            best_local_U, best_local_V, best_local_W = U_try.copy(), V_try.copy(), W_try.copy()
            
            for step in range(2000):
                if time.time() > deadline:
                    break
                
                # Compute gradients numerically (or analytically)
                # For speed, use finite differences on a subset
                eps = 1e-5
                
                # Compute current residual
                res = tensor_residual(U_try, V_try, W_try)
                
                if res < best_local_residual:
                    best_local_residual = res
                    best_local_U, best_local_V, best_local_W = U_try.copy(), V_try.copy(), W_try.copy()
                
                if res < 1e-10:
                    break
                
                # Compute gradient for U (finite difference)
                # For efficiency, only update one matrix at a time or use batch updates
                
                # Update U
                for idx in range(min(10, n*m)):  # Update a subset for speed
                    for r in range(target_rank):
                        pos = idx * target_rank + r
                        U_try[idx, r] += eps
                        res_plus = tensor_residual(U_try, V_try, W_try)
                        U_try[idx, r] -= 2*eps
                        res_minus = tensor_residual(U_try, V_try, W_try)
                        U_try[idx, r] += eps
                        grad = (res_plus - res_minus) / (2*eps)
                        U_try[idx, r] -= lr * grad
                
                # Update V
                for idx in range(min(10, m*p)):
                    for r in range(target_rank):
                        pos = idx * target_rank + r
                        V_try[idx, r] += eps
                        res_plus = tensor_residual(U_try, V_try, W_try)
                        V_try[idx, r] -= 2*eps
                        res_minus = tensor_residual(U_try, V_try, W_try)
                        V_try[idx, r] += eps
                        grad = (res_plus - res_minus) / (2*eps)
                        V_try[idx, r] -= lr * grad
                
                # Update W
                for idx in range(min(10, p*n)):
                    for r in range(target_rank):
                        pos = idx * target_rank + r
                        W_try[idx, r] += eps
                        res_plus = tensor_residual(U_try, V_try, W_try)
                        W_try[idx, r] -= 2*eps
                        res_minus = tensor_residual(U_try, V_try, W_try)
                        W_try[idx, r] += eps
                        grad = (res_plus - res_minus) / (2*eps)
                        W_try[idx, r] -= lr * grad
                
                # Adapt learning rate
                if res < best_local_residual * 0.9:
                    lr *= 1.1
                else:
                    lr *= 0.9
                lr = min(lr, 0.1)
            
            if best_local_residual < 1e-8:
                best_U, best_V, best_W = best_local_U, best_local_V, best_local_W
                best_rank = target_rank
                best_residual = best_local_residual
                found = True
                break
        
        if found:
            # Try to go lower
            continue
        else:
            # Can't find a valid decomposition at this rank, stop
            break
    
    # Final verification
    if best_residual < 1e-8:
        U_final = best_U
        V_final = best_V
        W_final = best_W
    else:
        # Fall back to schoolbook
        U_final = np.zeros((n*m, schoolbook_rank))
        V_final = np.zeros((m*p, schoolbook_rank))
        W_final = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_final[i*m+j, col] = 1.0
                    V_final[j*p+k, col] = 1.0
                    W_final[k*n+i, col] = 1.0
                    col += 1
    
    # Convert to list format
    def to_list(arr):
        return [[float(arr[i][j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    
    return {"U": to_list(U_final), "V": to_list(V_final), "W": to_list(W_final)}
# EVOLVE-BLOCK-END
