import numpy as np
from fractions import Fraction
import time
import sys

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 115  # leave margin before 120s kill
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Lower bound is max(n*m, m*p, n*p) = max(15, 35, 21) = 35
    # Try ranks from 35 upward
    
    np.random.seed(42)
    
    # Precompute index arrays for vectorized operations
    # For each (i,j,k), we need U[i*m+j,:], V[j*p+k,:], W[k*n+i,:]
    
    for target_rank in range(35, 106, 3):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(5):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Gradient descent with vectorized operations
            lr = 0.001
            for step in range(200):
                if time.time() - start_time > time_limit - 5:
                    break
                
                # Compute tensor values: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                # Vectorized: for each c, compute outer product structure
                
                # Build the full tensor efficiently
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                
                # For each c, the contribution is a rank-1 tensor in (i,j,k) space
                # U_c has shape (n*m,), V_c has shape (m*p,), W_c has shape (p*n,)
                
                # Reshape for computation
                U_r = U.reshape(n, m, target_rank)  # U[i,j,c]
                V_r = V.reshape(m, p, target_rank)  # V[j,k,c]
                W_r = W.reshape(p, n, target_rank)  # W[k,i,c]
                
                # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
                # This is a contraction over c
                
                # Compute T using einsum
                T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                
                # Error
                err = T - 1.0  # shape (n, m, p)
                
                # Gradients
                # dT/dc for U[i,j,c] = V_r[j,k,c] * W_r[k,i,c] summed over k
                # dT/dc for V[j,k,c] = U_r[i,j,c] * W_r[k,i,c] summed over i
                # dT/dc for W[k,i,c] = U_r[i,j,c] * V_r[j,k,c] summed over j
                
                # U_grad[i,j,c] = sum_k err[i,j,k] * V_r[j,k,c] * W_r[k,i,c]
                U_grad = np.einsum('ijk,jkc,kic->ijc', err, V_r, W_r)
                
                # V_grad[j,k,c] = sum_i err[i,j,k] * U_r[i,j,c] * W_r[k,i,c]
                V_grad = np.einsum('ijk,ijc,kic->jkc', err, U_r, W_r)
                
                # W_grad[k,i,c] = sum_j err[i,j,k] * U_r[i,j,c] * V_r[j,k,c]
                W_grad = np.einsum('ijk,ijc,jkc->kic', err, U_r, V_r)
                
                # Update
                U -= lr * U_grad
                V -= lr * V_grad
                W -= lr * W_grad
                
                # Check convergence
                if step % 50 == 49:
                    loss = np.sum(err**2)
                    if loss < 1e-8:
                        found = True
                        break
            
            if found:
                # Convert to list format
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                # Verify
                T_check = np.einsum('ijc,jkc,kic->ijk', 
                                   U.reshape(n,m,target_rank), 
                                   V.reshape(m,p,target_rank), 
                                   W.reshape(p,n,target_rank))
                if np.max(np.abs(T_check - 1.0)) < 1e-6:
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    break
        
        if best_rank < rank_sb:
            # Found something better, keep searching for even better
            continue
        else:
            # Didn't find improvement at this rank, try higher
            continue
    
    return best_result
