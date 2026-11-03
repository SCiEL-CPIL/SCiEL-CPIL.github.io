import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct Schoolbook Decomposition (Rank 84)
    def schoolbook(n, m, p):
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    U_school, V_school, W_school = schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n*m*p
    
    # 2. Helper: Compute residual tensor T_hat - 1
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # T_hat[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    def compute_residual(U, V, W):
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        T_hat = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        residual = T_hat - 1.0
        return residual, np.linalg.norm(residual)

    # 3. Helper: Snap to nearest 0.5 and check validity
    def snap_and_check(U, V, W, tol=1e-6):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        # Check if snapped solution is exact
        _, norm = compute_residual(U_s, V_s, W_s)
        if norm < tol:
            return U_s, V_s, W_s, norm
        return None

    # 4. Optimization Strategy: Gradient Descent with Snapping
    # We search for solutions with rank R < 84.
    # We use a "restart" strategy with random initializations.
    # To improve chances of finding exact half-integer solutions, we periodically snap and check.
    # We also try to prune terms (columns) that become small during optimization,
    # effectively searching lower ranks within a higher-rank frame, but since we fix R,
    # we just rely on the optimization to drive unused columns to zero (which snaps to 0).
    
    def try_optimize_snap(R, time_budget=2.0, seed=0):
        np.random.seed(seed)
        start_time = time.time()
        
        # Initialize with small random values to avoid symmetry breaking issues initially
        # Scaling by 0.1 helps with stability
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        lr = 0.05
        best_norm = np.inf
        
        for step in range(5000):
            if time.time() - start_time > time_budget:
                break
            
            residual, norm = compute_residual(U, V, W)
            
            if norm < best_norm:
                best_norm = norm
                # Save best state? Optional, but good for recovery
                # U_best = U.copy() ...
                
            # Check for convergence to a snapped solution
            if norm < 0.5: # Threshold for trying to snap
                snapped = snap_and_check(U, V, W)
                if snapped:
                    return snapped
            
            # Compute Gradients
            U_3d = U.reshape(n, m, R)
            V_3d = V.reshape(m, p, R)
            W_3d = W.reshape(p, n, R)
            
            # dU[i,j,r] = sum_k res[i,j,k] * V[j,k,r] * W[k,i,r]
            dU = np.einsum('ijk,jkr,kir->ijr', residual, V_3d, W_3d)
            # dV[j,k,r] = sum_i res[i,j,k] * U[i,j,r] * W[k,i,r]
            dV = np.einsum('ijk,ijr,kir->jkr', residual, U_3d, W_3d)
            # dW[k,i,r] = sum_j res[i,j,k] * U[i,j,r] * V[j,k,r]
            dW = np.einsum('ijk,ijr,jkr->kir', residual, U_3d, V_3d)
            
            # Update
            U -= lr * dU.reshape(n*m, R)
            V -= lr * dV.reshape(m*p, R)
            W -= lr * dW.reshape(p*n, R)
            
            # Adaptive learning rate decay
            lr *= 0.999
            
            # Optional: Zero out very small columns to encourage sparsity/rank reduction
            # This is a heuristic. If a column r is all zeros in U, V, W, it's useless.
            # We can explicitly zero out columns with small norms to help snapping.
            if step % 100 == 0:
                col_norms_U = np.linalg.norm(U, axis=0)
                col_norms_V = np.linalg.norm(V, axis=0)
                col_norms_W = np.linalg.norm(W, axis=0)
                
                # If a column is negligible in all three, zero it out completely
                mask = (col_norms_U < 1e-3) & (col_norms_V < 1e-3) & (col_norms_W < 1e-3)
                if np.any(mask):
                    U[:, mask] = 0
                    V[:, mask] = 0
                    W[:, mask] = 0

        # Final check
        snapped = snap_and_check(U, V, W)
        if snapped:
            return snapped
            
        return None

    # Search loop
    # We know 84 works.
    # We want to find R < 84.
    # We will iterate R from 83 down to 20.
    # For each R, we try a few seeds.
    # If we find a solution, we record it and continue to try lower R.
    # If time is up, we return the best found.
    
    search_start = time.time()
    max_search_time = 110.0 # Leave 10s margin for overhead and return
    
    # We prioritize lower ranks, but finding them is harder.
    # Starting from 83 and going down ensures we find "easy" improvements first.
    # However, the goal is rank-minimal.
    # Let's try a mix: some high ranks (easy), some low ranks (hard).
    # Or just go down.
    
    # Let's go down from 83 to 20.
    # To save time, we limit the number of seeds for each rank.
    # If we find a solution for R, we can stop trying that R.
    
    # We will keep track of the best rank found.
    best_rank_found = 84
    best_U_found, best_V_found, best_W_found = U_school, V_school, W_school
    
    # Ranks to try: 83, 82, ..., 20
    # We can skip some if time is tight, but let's start with all.
    # We will use a time budget per rank that decreases as we go lower?
    # No, just a fixed small budget per seed.
    
    for R in range(83, 19, -1):
        if time.time() - search_start > max_search_time - 5:
            break
            
        # Try a few seeds for this rank
        # We try 3 seeds for higher ranks (easier) and 5 for lower?
        # Let's stick to 3 seeds for now to cover more ranks.
        for seed in range(3):
            if time.time() - search_start > max_search_time - 2:
                break
                
            res = try_optimize_snap(R, time_budget=1.5, seed=seed)
            if res:
                U_sol, V_sol, W_sol, norm = res
                # Verify strictly
                _, final_norm = compute_residual(U_sol, V_sol, W_sol)
                if final_norm < 1e-5:
                    best_U_found, best_V_found, best_W_found = U_sol, V_sol, W_sol
                    best_rank_found = R
                    # We found a solution for R.
                    # We can break the seed loop for this R, as we have a valid solution.
                    # We continue to the next (lower) R.
                    break
                    
    # Return the best found
    U_out = best_U_found.tolist()
    V_out = best_V_found.tolist()
    W_out = best_W_found.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
