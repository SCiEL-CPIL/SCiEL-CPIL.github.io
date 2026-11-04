import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    U_sch = np.zeros((n*m, schoolbook_rank))
    V_sch = np.zeros((m*p, schoolbook_rank))
    W_sch = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1
                V_sch[j*p+k, col] = 1
                W_sch[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sch.tolist(), "V": V_sch.tolist(), "W": W_sch.tolist()}
    
    # Lower bound is max(n*m, m*p, p*n) = 15
    lower_bound = max(n*m, m*p, p*n)
    
    def check_exact(U, V, W, tol=1e-9):
        """Check if the decomposition is exact."""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)
    
    def try_rounding(U, V, W):
        """Try to round to nearest half-integer and check if still exact."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if check_exact(U_r, V_r, W_r):
            return U_r, V_r, W_r
        return None
    
    def init_by_merging(rank):
        """Initialize by merging schoolbook columns randomly."""
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        # Start with schoolbook and randomly merge columns
        for r in range(rank):
            # Pick 2-3 schoolbook columns to merge
            num_merge = min(3, schoolbook_rank - r)
            indices = random.sample(range(r, schoolbook_rank), num_merge)
            for idx in indices:
                U[:, r] += U_sch[:, idx]
                V[:, r] += V_sch[:, idx]
                W[:, r] += W_sch[:, idx]
        return U, V, W
    
    def init_by_partial_merge(rank):
        """Initialize with partial merges - some columns are single, some are merged."""
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        used = set()
        for r in range(rank):
            if random.random() < 0.3:
                # Single column
                while True:
                    idx = random.randint(0, schoolbook_rank - 1)
                    if idx not in used:
                        used.add(idx)
                        break
                U[:, r] = U_sch[:, idx]
                V[:, r] = V_sch[:, idx]
                W[:, r] = W_sch[:, idx]
            else:
                # Merge 2 columns
                while True:
                    idx1 = random.randint(0, schoolbook_rank - 1)
                    idx2 = random.randint(0, schoolbook_rank - 1)
                    if idx1 != idx2 and idx1 not in used and idx2 not in used:
                        used.add(idx1)
                        used.add(idx2)
                        break
                U[:, r] = U_sch[:, idx1] + U_sch[:, idx2]
                V[:, r] = V_sch[:, idx1] + V_sch[:, idx2]
                W[:, r] = W_sch[:, idx1] + W_sch[:, idx2]
        return U, V, W
    
    def lattice_search(rank, max_iters=3000, restarts=15):
        """
        Perform random walk on integer/half-integer lattice to find exact decomposition.
        """
        for restart in range(restarts):
            if time.time() > deadline - 5:
                return None
            
            # Choose initialization strategy
            init_method = random.choice(['random', 'merge', 'partial_merge'])
            if init_method == 'random':
                U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank))
                V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank))
                W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank))
            elif init_method == 'merge':
                U, V, W = init_by_merging(rank)
            else:
                U, V, W = init_by_partial_merge(rank)
            
            # Compute initial T_approx
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T - T_approx
            curr_error = np.sum(diff**2)
            
            if curr_error < 1e-10:
                rounded = try_rounding(U, V, W)
                if rounded:
                    return rounded
                return U, V, W
            
            temperature = 2.0
            cooling = 0.9995
            best_error = curr_error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Track non-zero diff indices for more efficient sampling
            nonzero_indices = np.argwhere(np.abs(diff) > 1e-10)
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random non-zero element in diff
                if len(nonzero_indices) == 0:
                    break
                    
                a, b, c = nonzero_indices[np.random.randint(len(nonzero_indices))]
                
                if abs(diff[a,b,c]) < 1e-10:
                    # Update nonzero list
                    nonzero_indices = np.argwhere(np.abs(diff) > 1e-10)
                    continue
                
                # Pick a random rank component r to modify
                r = np.random.randint(rank)
                
                # Decide which matrix to modify
                mod_choice = np.random.randint(3)
                
                old_val = 0.0
                new_val = 0.0
                delta = 0.0
                
                if mod_choice == 0:
                    old_val = U[a, r]
                    # Try +/- 0.5 or +/- 1, weighted toward smaller steps
                    step = np.random.choice([0.5, 1.0, 0.5, 0.5]) * np.random.choice([-1, 1])
                    new_val = old_val + step
                    delta = step
                elif mod_choice == 1:
                    old_val = V[b, r]
                    step = np.random.choice([0.5, 1.0, 0.5, 0.5]) * np.random.choice([-1, 1])
                    new_val = old_val + step
                    delta = step
                else:
                    old_val = W[c, r]
                    step = np.random.choice([0.5, 1.0, 0.5, 0.5]) * np.random.choice([-1, 1])
                    new_val = old_val + step
                    delta = step
                    
                # Calculate change in error
                if mod_choice == 0:
                    slice_old = diff[a]
                    update_vec = V[:, r] * W[:, r]
                    slice_new = slice_old - delta * update_vec
                    delta_err = np.sum(slice_new**2) - np.sum(slice_old**2)
                    
                elif mod_choice == 1:
                    slice_old = diff[:, b, :]
                    update_vec = U[:, r][:, np.newaxis] * W[:, r][np.newaxis, :]
                    slice_new = slice_old - delta * update_vec
                    delta_err = np.sum(slice_new**2) - np.sum(slice_old**2)
                    
                else:
                    slice_old = diff[:, :, c]
                    update_vec = U[:, r][:, np.newaxis] * V[:, r][np.newaxis, :]
                    slice_new = slice_old - delta * update_vec
                    delta_err = np.sum(slice_new**2) - np.sum(slice_old**2)

                # Accept if delta_err < 0 or with probability exp(-delta_err / temperature)
                accepted = False
                if delta_err < 0:
                    accepted = True
                else:
                    if temperature > 1e-6:
                        prob = np.exp(-delta_err / temperature)
                        if np.random.random() < prob:
                            accepted = True
                
                if accepted:
                    if mod_choice == 0:
                        U[a, r] = new_val
                        diff[a] -= delta * (V[:, r] * W[:, r])
                    elif mod_choice == 1:
                        V[b, r] = new_val
                        diff[:, b, :] -= delta * (U[:, r][:, np.newaxis] * W[:, r][np.newaxis, :])
                    else:
                        W[c, r] = new_val
                        diff[:, :, c] -= delta * (U[:, r][:, np.newaxis] * V[:, r][np.newaxis, :])
                    
                    curr_error += delta_err
                    if curr_error < best_error:
                        best_error = curr_error
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        
                    if curr_error < 1e-10:
                        rounded = try_rounding(U, V, W)
                        if rounded:
                            return rounded
                        return U, V, W
                        
                    # Update nonzero indices periodically
                    if it % 100 == 0:
                        nonzero_indices = np.argwhere(np.abs(diff) > 1e-10)
                            
                temperature *= cooling
                
            # End of iters for this restart
            # Try rounding the best found
            rounded = try_rounding(best_U, best_V, best_W)
            if rounded:
                return rounded
                
            if best_error < 1e-10:
                return best_U, best_V, best_W
                
        return None
    
    for target_rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
            
        # Try to find a solution for this rank
        # Use more restarts for lower ranks
        restarts = 20 if target_rank <= 20 else 10
        max_iters = 4000 if target_rank <= 20 else 2000
        
        result = lattice_search(target_rank, max_iters=max_iters, restarts=restarts)
        
        if result is not None:
            U, V, W = result
            # Final verification
            if check_exact(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return best_result
# EVOLVE-BLOCK-END
