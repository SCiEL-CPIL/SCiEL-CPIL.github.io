"""Combined approach: deterministic schoolbook fallback + structure-aware randomized search
with high-precision polish for rank-minimal 4x4 x 4x5 matrix multiplication tensor."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 140  # safety margin before 150s limit
    
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
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1
    
    # Randomized search with structure-aware initialization
    random.seed(42)
    np.random.seed(42)
    
    # Try ranks from lower values upward
    ranks_to_try = [60, 64, 68, 72, 76]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
        
        max_trials = 8 if r <= 68 else 4
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            # Structure-aware initialization: initialize based on the sparsity pattern
            # of the target tensor to give the solver a head start
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # For each rank component, assign non-zero entries based on the 
            # connectivity in the target tensor
            for rr in range(r):
                # Randomly select a "hub" index for each factor
                hub_a = np.random.randint(0, n*m)
                hub_b = np.random.randint(0, m*p)
                hub_c = np.random.randint(0, p*n)
                
                # Extract the i,j,k indices from the hubs
                i_a, j_a = divmod(hub_a, m)
                j_b, k_b = divmod(hub_b, p)
                k_c, i_c = divmod(hub_c, n)
                
                # For consistency, we want i_a == i_c and j_a == j_b
                # If not, we still proceed but with smaller weights
                weight = 1.0 if (i_a == i_c and j_a == j_b) else 0.3
                
                # Assign values with the structure in mind
                U[hub_a, rr] = weight * np.random.choice([-1, 1])
                V[hub_b, rr] = weight * np.random.choice([-1, 1])
                W[hub_c, rr] = weight * np.random.choice([-1, 1])
                
                # Add some additional non-zero entries to increase connectivity
                extra = np.random.randint(1, 4)
                for _ in range(extra):
                    idx_u = np.random.randint(0, n*m)
                    idx_v = np.random.randint(0, m*p)
                    idx_w = np.random.randint(0, p*n)
                    U[idx_u, rr] += 0.5 * np.random.choice([-1, 1])
                    V[idx_v, rr] += 0.5 * np.random.choice([-1, 1])
                    W[idx_w, rr] += 0.5 * np.random.choice([-1, 1])
            
            # Newton-Raphson with adaptive step size and high-precision polish
            lr_base = 0.05
            max_iter = 300
            min_res = np.inf
            stagnation_count = 0
            converged = False
            
            for step in range(max_iter):
                if time.time() > deadline - 3:
                    break
                
                # Compute current tensor estimate
                est = np.einsum('ar,br,cr->abc', U, V, W)
                residual = est - target
                max_res = np.max(np.abs(residual))
                
                if max_res < 1e-12:
                    converged = True
                    break
                
                if max_res < min_res:
                    min_res = max_res
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                
                if stagnation_count > 30:
                    # Perturb to escape local minimum
                    U += np.random.randn(n*m, r) * 0.05
                    V += np.random.randn(m*p, r) * 0.05
                    W += np.random.randn(p*n, r) * 0.05
                    stagnation_count = 0
                    continue
                
                # Compute gradients
                dU = np.einsum('abc,br,cr->ar', residual, V, W)
                dV = np.einsum('abc,ar,cr->br', residual, U, W)
                dW = np.einsum('abc,ar,br->cr', residual, U, V)
                
                # Adaptive learning rate with decay
                lr = lr_base / (1.0 + 0.02 * step)
                
                # Backtracking line search
                for backtrack in range(8):
                    U_new = U - lr * dU
                    V_new = V - lr * dV
                    W_new = W - lr * dW
                    
                    est_new = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                    res_new = np.max(np.abs(est_new - target))
                    
                    if res_new < max_res:
                        U, V, W = U_new, V_new, W_new
                        break
                    else:
                        lr *= 0.5
            
            # High-precision polish pass if close to convergence
            est_final = np.einsum('ar,br,cr->abc', U, V, W)
            final_res = np.max(np.abs(est_final - target))
            
            if final_res < 1e-6:
                # Polish with tighter tolerance
                for polish_step in range(100):
                    if time.time() > deadline - 2:
                        break
                    
                    est = np.einsum('ar,br,cr->abc', U, V, W)
                    residual = est - target
                    max_res = np.max(np.abs(residual))
                    
                    if max_res < 1e-14:
                        break
                    
                    dU = np.einsum('abc,br,cr->ar', residual, V, W)
                    dV = np.einsum('abc,ar,cr->br', residual, U, W)
                    dW = np.einsum('abc,ar,br->cr', residual, U, V)
                    
                    lr = 0.01 / (1.0 + 0.01 * polish_step)
                    
                    for backtrack in range(6):
                        U_new = U - lr * dU
                        V_new = V - lr * dV
                        W_new = W - lr * dW
                        
                        est_new = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                        res_new = np.max(np.abs(est_new - target))
                        
                        if res_new < max_res:
                            U, V, W = U_new, V_new, W_new
                            break
                        else:
                            lr *= 0.5
                
                est_polished = np.einsum('ar,br,cr->abc', U, V, W)
                if np.max(np.abs(est_polished - target)) < 1e-10:
                    # Try to clean up to integers/half-integers
                    best_clean = None
                    best_clean_err = np.inf
                    
                    for factor in [1, 2, 4, 8, 16]:
                        U_clean = np.round(U * factor) / factor
                        V_clean = np.round(V * factor) / factor
                        W_clean = np.round(W * factor) / factor
                        
                        est_clean = np.einsum('ar,br,cr->abc', U_clean, V_clean, W_clean)
                        err = np.max(np.abs(est_clean - target))
                        
                        if err < 1e-8 and err < best_clean_err:
                            best_clean = (U_clean, V_clean, W_clean)
                            best_clean_err = err
                    
                    if best_clean is not None:
                        best_rank = r
                        U_c, V_c, W_c = best_clean
                        best_result = {"U": U_c.tolist(), "V": V_c.tolist(), "W": W_c.tolist()}
                        break
                    else:
                        # Use the float version
                        best_rank = r
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        break
        
        if best_rank < n * m * p:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
