import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = start_time + 110.0

    # Try to find a lower rank decomposition
    # Known lower bounds suggest rank is around 20-30.
    # We try ranks from 25 to 40.
    target_ranks = [25, 26, 27, 28, 29, 30, 32, 34, 36, 38, 40]
    
    best_result = None
    best_rank = rank_sb

    for target_rank in target_ranks:
        if target_rank >= rank_sb:
            continue
        if time.time() > deadline - 10:
            break
            
        # Try multiple restarts for this rank
        found = False
        restarts = 20 # Adjust based on time
        for r in range(restarts):
            if time.time() > deadline - 5:
                break
            result = search_rank_cd(n, m, p, target_rank, deadline)
            if result is not None:
                return result
            if result == "improved":
                 # We found a good candidate but not exact? 
                 # The function should return exact or None.
                 pass

    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def search_rank_cd(n, m, p, rank, deadline):
    """
    Uses discrete coordinate descent to find an exact half-integer decomposition.
    """
    # Initialize with small random half-integers
    # Using integers 0, 1, -1, 0.5, -0.5 etc.
    # Let's stick to integers first, then half integers.
    # Random init in {-1, 0, 1}
    U = np.random.randint(-1, 2, size=(n * m, rank))
    V = np.random.randint(-1, 2, size=(m * p, rank))
    W = np.random.randint(-1, 2, size=(p * n, rank))
    
    # Reshape for easier access
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Precompute indices for W access in T
    # T[i, j, k] = sum_r U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
    
    # To speed up coordinate descent, we maintain the current tensor T.
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i, j, k] = dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
                # This is O(rank)
                T[i, j, k] = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
                
    # Error = sum (T - 1)^2
    current_error = np.sum((T - 1.0)**2)
    
    # Coordinate Descent Loop
    # We iterate over all variables, trying to change them to nearby half-integers.
    # Neighbors: current, current +/- 0.5, current +/- 1.0
    # Since we start with integers, we can try half-integers.
    
    max_iters = 5000 # Epochs of full passes
    no_improve_count = 0
    threshold = 10
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
            
        improved = False
        # Randomize order to escape local minima
        # Create a list of all variable indices
        # U vars: (0, i, j, r)
        # V vars: (1, j, k, r)
        # W vars: (2, k, i, r)
        
        # For efficiency, we just loop through them in random order
        # Total vars = n*m*rank + m*p*rank + p*n*rank
        num_u = n * m * rank
        num_v = m * p * rank
        num_w = p * n * rank
        total_vars = num_u + num_v + num_w
        
        # Instead of shuffling a huge list, we pick random indices
        # Or we do a fixed random permutation each epoch?
        # Let's just do a few random updates per epoch to save time?
        # No, full pass is better for convergence.
        
        # To make it faster, we only try 1 change per variable per epoch.
        
        # --- Update U ---
        for i in range(n):
            for j in range(m):
                for r in range(rank):
                    old_val = U3[i, j, r]
                    # Try neighbors
                    # The change in T for all (j, k) where U[i, j, r] is involved
                    # T[i, j, k] depends on U[i, j, r].
                    # Specifically, T[i, j, k] += delta * V3[j, k, r] * W3[k, i, r]
                    
                    # We want to minimize sum_k (T[i, j, k] + delta * V3[j, k, r] * W3[k, i, r] - 1)^2
                    # This is a quadratic in delta.
                    # Let A_k = V3[j, k, r] * W3[k, i, r]
                    # Let B_k = T[i, j, k] - 1
                    # Minimize sum_k (B_k + delta * A_k)^2
                    # Derivative: sum_k 2 (B_k + delta A_k) A_k = 0
                    # delta * sum(A_k^2) = - sum(B_k A_k)
                    # delta = - sum(B_k A_k) / sum(A_k^2)
                    
                    # However, we are constrained to half-integers.
                    # So we calculate the optimal real delta, then snap to the nearest half-integer neighbor.
                    
                    A = V3[j, :, r] * W3[:, i, r] # Shape (p,)
                    B = T[i, j, :] - 1.0           # Shape (p,)
                    
                    denom = np.dot(A, A)
                    if denom > 1e-9:
                        delta_opt = -np.dot(B, A) / denom
                        # Candidate values: old_val + delta_opt snapped to 0.5 grid
                        # Also try old_val +/- 0.5, +/- 1.0
                        candidates = [old_val]
                        # Snap delta_opt to 0.5 grid relative to old_val
                        # new_val = old_val + round(delta_opt * 2) / 2
                        cand_snap = old_val + np.round(delta_opt * 2) / 2
                        candidates.append(cand_snap)
                        candidates.append(cand_snap + 0.5)
                        candidates.append(cand_snap - 0.5)
                        candidates.append(old_val + 0.5)
                        candidates.append(old_val - 0.5)
                        candidates.append(old_val + 1.0)
                        candidates.append(old_val - 1.0)
                        
                        best_delta = 0
                        best_err = current_error # Current error contribution for this (i,j) slice?
                        # Actually, we need to compare the *change* in error.
                        # Let's compute the new error for the (i, j, :) slice for each candidate.
                        
                        slice_error_old = np.sum(B**2)
                        
                        best_cand = old_val
                        best_slice_error = slice_error_old
                        
                        for cand in candidates:
                            if cand == old_val:
                                continue
                            # Check if cand is a valid half-integer
                            if abs(cand * 2 - round(cand * 2)) > 1e-9:
                                continue
                                
                            delta = cand - old_val
                            # New B_k = B_k + delta * A_k
                            new_B = B + delta * A
                            new_slice_error = np.sum(new_B**2)
                            
                            if new_slice_error < best_slice_error - 1e-9:
                                best_slice_error = new_slice_error
                                best_cand = cand
                                
                        if best_cand != old_val:
                            delta_applied = best_cand - old_val
                            # Update T
                            T[i, j, :] += delta_applied * A
                            # Update U
                            U3[i, j, r] = best_cand
                            # Update global error
                            current_error += best_slice_error - slice_error_old
                            improved = True
                            
        # --- Update V ---
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    old_val = V3[j, k, r]
                    # T[i, j, k] depends on V[j, k, r]
                    # T[i, j, k] += delta * U3[i, j, r] * W3[k, i, r]
                    # A_i = U3[i, j, r] * W3[k, i, r] (for all i)
                    # B_i = T[:, j, k] - 1
                    
                    A = U3[:, j, r] * W3[k, :, r] # Shape (n,)
                    B = T[:, j, k] - 1.0           # Shape (n,)
                    
                    denom = np.dot(A, A)
                    if denom > 1e-9:
                        delta_opt = -np.dot(B, A) / denom
                        candidates = [old_val]
                        cand_snap = old_val + np.round(delta_opt * 2) / 2
                        candidates.append(cand_snap)
                        candidates.append(cand_snap + 0.5)
                        candidates.append(cand_snap - 0.5)
                        candidates.append(old_val + 0.5)
                        candidates.append(old_val - 0.5)
                        candidates.append(old_val + 1.0)
                        candidates.append(old_val - 1.0)
                        
                        slice_error_old = np.sum(B**2)
                        best_cand = old_val
                        best_slice_error = slice_error_old
                        
                        for cand in candidates:
                            if cand == old_val:
                                continue
                            if abs(cand * 2 - round(cand * 2)) > 1e-9:
                                continue
                            
                            delta = cand - old_val
                            new_B = B + delta * A
                            new_slice_error = np.sum(new_B**2)
                            
                            if new_slice_error < best_slice_error - 1e-9:
                                best_slice_error = new_slice_error
                                best_cand = cand
                                
                        if best_cand != old_val:
                            delta_applied = best_cand - old_val
                            T[:, j, k] += delta_applied * A
                            V3[j, k, r] = best_cand
                            current_error += best_slice_error - slice_error_old
                            improved = True
                            
        # --- Update W ---
        for k in range(p):
            for i in range(n):
                for r in range(rank):
                    old_val = W3[k, i, r]
                    # T[i, j, k] depends on W[k, i, r]
                    # T[i, j, k] += delta * U3[i, j, r] * V3[j, k, r]
                    # A_j = U3[i, j, r] * V3[j, k, r] (for all j)
                    # B_j = T[i, :, k] - 1
                    
                    A = U3[i, :, r] * V3[:, k, r] # Shape (m,)
                    B = T[i, :, k] - 1.0           # Shape (m,)
                    
                    denom = np.dot(A, A)
                    if denom > 1e-9:
                        delta_opt = -np.dot(B, A) / denom
                        candidates = [old_val]
                        cand_snap = old_val + np.round(delta_opt * 2) / 2
                        candidates.append(cand_snap)
                        candidates.append(cand_snap + 0.5)
                        candidates.append(cand_snap - 0.5)
                        candidates.append(old_val + 0.5)
                        candidates.append(old_val - 0.5)
                        candidates.append(old_val + 1.0)
                        candidates.append(old_val - 1.0)
                        
                        slice_error_old = np.sum(B**2)
                        best_cand = old_val
                        best_slice_error = slice_error_old
                        
                        for cand in candidates:
                            if cand == old_val:
                                continue
                            if abs(cand * 2 - round(cand * 2)) > 1e-9:
                                continue
                            
                            delta = cand - old_val
                            new_B = B + delta * A
                            new_slice_error = np.sum(new_B**2)
                            
                            if new_slice_error < best_slice_error - 1e-9:
                                best_slice_error = new_slice_error
                                best_cand = cand
                                
                        if best_cand != old_val:
                            delta_applied = best_cand - old_val
                            T[i, :, k] += delta_applied * A
                            W3[k, i, r] = best_cand
                            current_error += best_slice_error - slice_error_old
                            improved = True
                            
        if not improved:
            no_improve_count += 1
            if no_improve_count > threshold:
                # Try a random kick to escape local minimum
                # Perturb a few random variables
                for _ in range(10):
                    mat_idx = np.random.randint(0, 3)
                    if mat_idx == 0:
                        i, j, r = np.random.randint(0, n), np.random.randint(0, m), np.random.randint(0, rank)
                        old_val = U3[i, j, r]
                        # Pick a random neighbor
                        new_val = old_val + np.random.choice([-1.0, -0.5, 0.5, 1.0])
                        # Update T and Error
                        A = V3[j, :, r] * W3[:, i, r]
                        B = T[i, j, :] - 1.0
                        slice_error_old = np.sum(B**2)
                        delta = new_val - old_val
                        new_B = B + delta * A
                        new_slice_error = np.sum(new_B**2)
                        
                        T[i, j, :] += delta * A
                        U3[i, j, r] = new_val
                        current_error += new_slice_error - slice_error_old
                    elif mat_idx == 1:
                        j, k, r = np.random.randint(0, m), np.random.randint(0, p), np.random.randint(0, rank)
                        old_val = V3[j, k, r]
                        new_val = old_val + np.random.choice([-1.0, -0.5, 0.5, 1.0])
                        A = U3[:, j, r] * W3[k, :, r]
                        B = T[:, j, k] - 1.0
                        slice_error_old = np.sum(B**2)
                        delta = new_val - old_val
                        new_B = B + delta * A
                        new_slice_error = np.sum(new_B**2)
                        
                        T[:, j, k] += delta * A
                        V3[j, k, r] = new_val
                        current_error += new_slice_error - slice_error_old
                    else:
                        k, i, r = np.random.randint(0, p), np.random.randint(0, n), np.random.randint(0, rank)
                        old_val = W3[k, i, r]
                        new_val = old_val + np.random.choice([-1.0, -0.5, 0.5, 1.0])
                        A = U3[i, :, r] * V3[:, k, r]
                        B = T[i, :, k] - 1.0
                        slice_error_old = np.sum(B**2)
                        delta = new_val - old_val
                        new_B = B + delta * A
                        new_slice_error = np.sum(new_B**2)
                        
                        T[i, :, k] += delta * A
                        W3[k, i, r] = new_val
                        current_error += new_slice_error - slice_error_old
                        
            no_improve_count = 0 # Reset after kick? Or keep counting?
            # Let's reset to allow more kicks if stuck
            if no_improve_count > 100:
                break
                
        if current_error < 1e-10:
            # Found exact solution
            return {
                "U": U3.reshape(n*m, rank).tolist(),
                "V": V3.reshape(m*p, rank).tolist(),
                "W": W3.reshape(p*n, rank).tolist()
            }
            
    return None

def compute_error(U, V, W, n, m, p):
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    error = np.sum((T - 1.0)**2)
    return error
