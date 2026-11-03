"""Combined approach: deterministic schoolbook fallback + randomized Newton-Raphson
polynomial system solver for rank-minimal 4x4 x 4x5 matrix multiplication tensor."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 145  # safety margin before 150s limit
    
    # Schoolbook fallback: rank 80
    best_rank = n * m * p  # 80
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Build target tensor
    # T[a, b, c] = 1 if (a,b,c) corresponds to valid (i,j,k), else 0
    # a = i*m + j, b = j*p + k, c = k*n + i
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1
    
    # The decomposition requires:
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all a,b,c
    # This is a system of polynomial equations in the variables U, V, W.
    
    # Randomized Newton-Raphson approach:
    # Initialize with Gaussian noise, then iterate:
    # 1. Compute residual R = T - einsum(U,V,W)
    # 2. Compute Jacobian-like update (using first-order approximation)
    # 3. Update U, V, W
    
    random.seed(12345)
    np.random.seed(12345)
    
    # Try ranks from lower values upward
    # For 4x4x4, Laderman gives rank 49. For 4x4x5, try from 60 upward
    ranks_to_try = [60, 62, 64, 66, 68, 70, 72, 74, 76, 78]
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            continue
        
        # Try multiple random initializations
        max_trials = 8 if r <= 68 else 4
        
        for trial in range(max_trials):
            if time.time() > deadline - 3:
                break
            
            # Random initialization with small variance
            U = np.random.randn(n*m, r) * 0.3
            V = np.random.randn(m*p, r) * 0.3
            W = np.random.randn(p*n, r) * 0.3
            
            # Newton-Raphson iteration
            # The equations are: F[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c] = 0
            # We use a damped Newton method with line search
            
            lr_base = 0.05
            max_iter = 300
            converged = False
            
            for step in range(max_iter):
                if time.time() > deadline - 2:
                    break
                
                # Compute current tensor estimate
                est = np.einsum('ar,br,cr->abc', U, V, W)
                residual = est - target
                max_res = np.max(np.abs(residual))
                
                if max_res < 1e-10:
                    converged = True
                    break
                
                # Compute gradients (Jacobian applied to identity)
                # dF/dU[a,r] = sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
                # dF/dV[b,r] = sum_{a,c} residual[a,b,c] * U[a,r] * W[c,r]
                # dF/dW[c,r] = sum_{a,b} residual[a,b,c] * U[a,r] * V[b,r]
                
                dU = np.einsum('abc,br,cr->ar', residual, V, W)
                dV = np.einsum('abc,ar,cr->br', residual, U, W)
                dW = np.einsum('abc,ar,br->cr', residual, U, V)
                
                # Adaptive learning rate with damping
                lr = lr_base / (1.0 + 0.1 * step)
                
                # Line search: try to reduce the residual
                # First, try full step
                U_new = U - lr * dU
                V_new = V - lr * dV
                W_new = W - lr * dW
                
                est_new = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                res_new = np.max(np.abs(est_new - target))
                
                if res_new < max_res:
                    U, V, W = U_new, V_new, W_new
                else:
                    # Halve learning rate
                    lr /= 2.0
                    U_new = U - lr * dU
                    V_new = V - lr * dV
                    W_new = W - lr * dW
                    est_new = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                    res_new = np.max(np.abs(est_new - target))
                    if res_new < max_res:
                        U, V, W = U_new, V_new, W_new
                    else:
                        # Try even smaller step
                        lr /= 2.0
                        U_new = U - lr * dU
                        V_new = V - lr * dV
                        W_new = W - lr * dW
                        est_new = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                        res_new = np.max(np.abs(est_new - target))
                        if res_new < max_res:
                            U, V, W = U_new, V_new, W_new
                        else:
                            # Stagnation - try a random perturbation
                            U += np.random.randn(n*m, r) * 0.01
                            V += np.random.randn(m*p, r) * 0.01
                            W += np.random.randn(p*n, r) * 0.01
                
                # Check for convergence
                if step % 50 == 49:
                    est_check = np.einsum('ar,br,cr->abc', U, V, W)
                    if np.max(np.abs(est_check - target)) < 1e-9:
                        converged = True
                        break
            
            # Final verification
            est_final = np.einsum('ar,br,cr->abc', U, V, W)
            if np.max(np.abs(est_final - target)) < 1e-8:
                # Found a valid decomposition!
                # Try to clean up to integers/half-integers
                U_clean = np.round(U * 2) / 2
                V_clean = np.round(V * 2) / 2
                W_clean = np.round(W * 2) / 2
                
                est_clean = np.einsum('ar,br,cr->abc', U_clean, V_clean, W_clean)
                if np.max(np.abs(est_clean - target)) < 1e-7:
                    best_rank = r
                    best_result = {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}
                    break
                else:
                    # Use the float version if it's valid
                    best_rank = r
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
        
        if best_rank < n * m * p:
            # Found improvement, try to go lower
            continue
    
    return best_result
# EVOLVE-BLOCK-END
