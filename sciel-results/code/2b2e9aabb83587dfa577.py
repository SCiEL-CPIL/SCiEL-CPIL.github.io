"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 40
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 130  # 20s margin for 150s limit
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # All other entries are 0.
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_target[idx_u, idx_v, idx_w] = 1.0
    
    # Helper to check if a decomposition is valid
    def is_valid(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the tensor."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        # Compute tensor product sum
        T_rec = np.zeros_like(T_target)
        for c in range(U.shape[1]):
            T_rec += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.allclose(T_rec, T_target, atol=tol, rtol=0)
    
    def check_half_integer(U, V, W, tol=1e-9):
        """Check if all entries are integer or half-integer."""
        for mat in [U, V, W]:
            for val in mat.flatten():
                v2 = val * 2
                if abs(v2 - round(v2)) > tol:
                    return False
        return True
    
    def round_to_half_integer(arr):
        """Round array to nearest half-integer."""
        return np.round(arr * 2) / 2.0
    
    # Schoolbook as fallback
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U_school[idx_u, col] = 1.0
                V_school[idx_v, col] = 1.0
                W_school[idx_w, col] = 1.0
                col += 1
    
    best_result = (U_school, V_school, W_school)
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions using iterative refinement
    # Strategy: Start from schoolbook, try to merge terms by finding
    # linear dependencies and using gradient-based refinement
    
    for target_rank in range(rank_schoolbook - 1, 24, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        num_restarts = 3 if target_rank > 30 else 5
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            # Initialize with random small values
            scale = 0.5
            U = np.random.randn(n * m, target_rank) * scale
            V = np.random.randn(m * p, target_rank) * scale
            W = np.random.randn(p * n, target_rank) * scale
            
            # Iterative refinement: alternate between updating each factor
            # using least squares given the other two factors
            converged = False
            max_iters = 200
            for it in range(max_iters):
                if time.time() > deadline - 15:
                    break
                
                # Update U: for each column c, minimize ||T - sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]||
                # This is equivalent to: for each (i,j), find U[i*m+j, :] such that
                # sum_c U[i*m+j, c] * (V[:,c] ⊗ W[:,c]) matches T[i*m+j, :, :]
                
                # Build the linear system for U
                # For fixed (i,j), we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i*m+j, j*p+k, k*n+i]
                # This is: U[i*m+j, :] @ (V[j*p+k, :] * W[k*n+i, :]) = T[i*m+j, j*p+k, k*n+i]
                
                # We can solve for all (i,j) simultaneously using least squares
                # Build matrix A where A[ab, c] = sum over valid (k) of V[j*p+k, c] * W[k*n+i, c]
                # But this is tricky because the mapping is not linear in a simple way.
                
                # Alternative: use gradient descent on the objective
                # f(U,V,W) = ||sum_c U[:,c]⊗V[:,c]⊗W[:,c] - T||_F^2
                
                # Compute gradient with respect to U
                # df/du[i,j,c] = 2 * sum_{(a,b) where a=i*m+j} (rec[a,b,:] - T[a,b,:]) * V[b,c] * W[:,c]... 
                # This is complex. Let's use a simpler approach.
                
                # Use ALS: fix V and W, solve for U
                # For each (i,j), we have: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i*m+j, j*p+k, k*n+i]
                # for all k. This is a linear system in U[i*m+j, :].
                
                # Build the system: for fixed (i,j), let A[i,j] be a matrix where
                # A[i,j][k, c] = V[j*p+k, c] * W[k*n+i, c]
                # Then U[i*m+j, :] should satisfy A[i,j] @ U[i*m+j, :] = T[i*m+j, j*p+0:(j+1)*p, k*n+i for k in 0..p-1]
                
                # Actually, let's just use a simple gradient step
                # Compute current reconstruction
                T_rec = np.zeros_like(T_target)
                for c in range(target_rank):
                    T_rec += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                
                # Compute error
                err = T_rec - T_target
                loss = np.sum(err ** 2)
                
                if loss < 1e-10:
                    converged = True
                    break
                
                # Compute gradients
                # dU[i,j,c] = 2 * sum over (k) err[i*m+j, j*p+k, k*n+i] * V[j*p+k, c] * W[k*n+i, c]
                # But this only accounts for the terms where (i,j,k) is valid.
                # For other terms, we need to account for them too.
                
                # More carefully: T_rec[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
                # dT_rec[a,b,c]/dU[a',d] = V[b,d] * W[c,d] if a=a', else 0
                # So dL/dU[a,d] = 2 * sum_{b,c} err[a,b,c] * V[b,d] * W[c,d]
                
                # This is expensive. Let's use a simpler heuristic:
                # Just do a small random perturbation and keep if loss decreases
                
                # Random perturbation
                lr = 0.01
                U_pert = U + lr * np.random.randn(*U.shape)
                V_pert = V + lr * np.random.randn(*V.shape)
                W_pert = W + lr * np.random.randn(*W.shape)
                
                T_rec_pert = np.zeros_like(T_target)
                for c in range(target_rank):
                    T_rec_pert += np.einsum('i,j,k->ijk', U_pert[:, c], V_pert[:, c], W_pert[:, c])
                
                err_pert = T_rec_pert - T_target
                loss_pert = np.sum(err_pert ** 2)
                
                if loss_pert < loss:
                    U, V, W = U_pert, V_pert, W_pert
                else:
                    # Try smaller step
                    lr = 0.005
                    U_pert = U + lr * np.random.randn(*U.shape)
                    V_pert = V + lr * np.random.randn(*V.shape)
                    W_pert = W + lr * np.random.randn(*W.shape)
                    
                    T_rec_pert = np.zeros_like(T_target)
                    for c in range(target_rank):
                        T_rec_pert += np.einsum('i,j,k->ijk', U_pert[:, c], V_pert[:, c], W_pert[:, c])
                    
                    err_pert = T_rec_pert - T_target
                    loss_pert = np.sum(err_pert ** 2)
                    
                    if loss_pert < loss:
                        U, V, W = U_pert, V_pert, W_pert
                    else:
                        break  # No progress, stop
            
            # Check if we found a valid decomposition
            if is_valid(U, V, W, tol=1e-6):
                # Try to round to half-integers
                U_rounded = round_to_half_integer(U)
                V_rounded = round_to_half_integer(V)
                W_rounded = round_to_half_integer(W)
                
                if is_valid(U_rounded, V_rounded, W_rounded, tol=1e-9):
                    best_result = (U_rounded, V_rounded, W_rounded)
                    best_rank = target_rank
                    found = True
                    break
                else:
                    # Check if the unrounded version is valid with tighter tolerance
                    if is_valid(U, V, W, tol=1e-9):
                        best_result = (U, V, W)
                        best_rank = target_rank
                        found = True
                        break
        
        if found:
            break  # Found a valid decomposition at this rank
    
    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
