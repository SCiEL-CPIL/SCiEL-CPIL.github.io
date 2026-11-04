import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def get_schoolbook():
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
        return {"U": U, "V": V, "W": W}, rank

    schoolbook_sol, schoolbook_rank = get_schoolbook()
    best_solution = schoolbook_sol
    best_rank = schoolbook_rank

    def compute_reconstruction(U, V, W, rank):
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return Rec

    def is_valid(U, V, W, rank, atol=1e-9):
        Rec = compute_reconstruction(U, V, W, rank)
        return np.allclose(Rec, 1.0, atol=atol)

    def local_optimize_fast(U, V, W, rank, max_iters=500, atol=1e-9):
        """
        Fast local optimization using incremental error updates.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Precompute valid values
        valid_vals = np.array([x / 2.0 for x in range(-10, 11)])
        
        # Initial reconstruction
        Rec = compute_reconstruction(U, V, W, rank)
        current_err = np.max(np.abs(Rec - 1.0))
        
        if current_err < atol:
            return U, V, W, True

        # Helper to update Rec and error when changing a single entry
        # U[i, r] changes Rec[i, :, :]
        # V[j, r] changes Rec[:, j, :]
        # W[k, r] changes Rec[:, :, k]
        
        # We store Rec as (n, m, p)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            if current_err < atol:
                break
            
            improved = False
            # Randomly select an entry to modify to avoid getting stuck in local minima
            # Or iterate through all? For small rank, iterating all is feasible.
            # Let's iterate through all non-zero or random subset.
            
            # To make it faster, we pick a random element from U, V, or W
            # and try to improve it.
            
            mat_type = np.random.randint(3) # 0: U, 1: V, 2: W
            r = np.random.randint(rank)
            
            if mat_type == 0:
                idx = np.random.randint(n * m)
                i = idx // m
                j = idx % m
                old_val = U[idx, r]
                
                # Try neighbors
                best_val = old_val
                best_err = current_err
                
                # Candidates: old_val, old_val +/- 0.5, old_val +/- 1.0
                # Ensure they are in valid_vals
                candidates = [old_val]
                for delta in [-1.0, -0.5, 0.5, 1.0]:
                    c = old_val + delta
                    if c in valid_vals:
                        candidates.append(c)
                
                for cand in candidates:
                    if cand == old_val:
                        continue
                    
                    # Update Rec
                    # delta = cand - old_val
                    delta = cand - old_val
                    # Rec[i, :, :] += delta * V[:, r].reshape(m, p) * W[:, r].reshape(p, n)
                    # Wait, Rec[i, j, k] += U[i, r] * V[j, r] * W[k, r]
                    # So Rec[i, j, k] changes by delta * V[j, r] * W[k, r]
                    
                    v_col = V[:, r]
                    w_col = W[:, r]
                    # Outer product of v_col and w_col is (m, p) x (p, n) -> (m, n)?
                    # No. V is (m, p) indexed by j, k? No, V[:, r] is length m*p.
                    # V[j*p+k, r] is the value.
                    # W[k*n+i, r] is the value.
                    # Rec[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    
                    # So for fixed i, r:
                    # Rec[i, j, k] += delta * V[j*p+k, r] * W[k*n+i, r]
                    
                    # Let's construct the update tensor for Rec[i, :, :]
                    # It is delta * (V_col reshaped (m, p)) outer (W_col reshaped (p, n)[:, i])?
                    # V_col is (m, p). W_col is (p, n).
                    # We need V[j, k] * W[k, i].
                    # This is (V_col * W_col[:, i]) summed over k? No, it's a product for each j, k.
                    # Rec[i, j, k] += delta * V[j, k] * W[k, i]
                    
                    V_mat = V[:, r].reshape(m, p)
                    W_mat = W[:, r].reshape(p, n)
                    
                    # Update slice Rec[i, :, :]
                    # The change is delta * (V_mat * W_mat[:, i])
                    # V_mat is (m, p), W_mat[:, i] is (p,).
                    # Broadcasting: V_mat * W_mat[:, i] -> (m, p)
                    
                    update_slice = delta * (V_mat * W_mat[:, i])
                    
                    old_slice = Rec[i, :, :].copy()
                    Rec[i, :, :] += update_slice
                    
                    new_err = np.max(np.abs(Rec - 1.0))
                    
                    if new_err < best_err - 1e-10:
                        best_err = new_err
                        best_val = cand
                    else:
                        Rec[i, :, :] = old_slice
                        # If we didn't accept, we need to restore Rec?
                        # We only update Rec if we accept?
                        # It's better to compute new_err and if it's better, keep the change in Rec and U.
                        # If not, revert Rec.
                
                if best_val != old_val:
                    U[idx, r] = best_val
                    current_err = best_err
                    improved = True
                    # Rec is already updated if we accepted?
                    # In the loop above, if we accept, we don't revert.
                    # But we only revert if we don't accept.
                    # So if best_val != old_val, Rec is updated.
            
            elif mat_type == 1:
                idx = np.random.randint(m * p)
                j = idx // p
                k = idx % p
                old_val = V[idx, r]
                
                best_val = old_val
                best_err = current_err
                
                candidates = [old_val]
                for delta in [-1.0, -0.5, 0.5, 1.0]:
                    c = old_val + delta
                    if c in valid_vals:
                        candidates.append(c)
                
                for cand in candidates:
                    if cand == old_val:
                        continue
                    
                    delta = cand - old_val
                    # Rec[:, j, k] += delta * U[i, r] * W[k, i]
                    # Rec[i, j, k] += U[i, r] * delta * W[k, i]
                    
                    U_mat = U[:, r].reshape(n, m)
                    W_mat = W[:, r].reshape(p, n)
                    
                    # Update slice Rec[:, j, k]
                    # Change is delta * U_mat[:, j] * W_mat[k, :]
                    # U_mat[:, j] is (n,), W_mat[k, :] is (n,).
                    # Product is (n,).
                    
                    update_slice = delta * (U_mat[:, j] * W_mat[k, :])
                    
                    old_slice = Rec[:, j, k].copy()
                    Rec[:, j, k] += update_slice
                    
                    new_err = np.max(np.abs(Rec - 1.0))
                    
                    if new_err < best_err - 1e-10:
                        best_err = new_err
                        best_val = cand
                    else:
                        Rec[:, j, k] = old_slice
                
                if best_val != old_val:
                    V[idx, r] = best_val
                    current_err = best_err
                    improved = True
            
            else: # mat_type == 2
                idx = np.random.randint(p * n)
                k = idx // n
                i = idx % n
                old_val = W[idx, r]
                
                best_val = old_val
                best_err = current_err
                
                candidates = [old_val]
                for delta in [-1.0, -0.5, 0.5, 1.0]:
                    c = old_val + delta
                    if c in valid_vals:
                        candidates.append(c)
                
                for cand in candidates:
                    if cand == old_val:
                        continue
                    
                    delta = cand - old_val
                    # Rec[i, :, k] += delta * U[i, r] * V[j, r]
                    # Rec[i, j, k] += U[i, r] * V[j, r] * delta
                    
                    U_mat = U[:, r].reshape(n, m)
                    V_mat = V[:, r].reshape(m, p)
                    
                    # Update slice Rec[i, :, k]
                    # Change is delta * U_mat[i, :] * V_mat[:, k]
                    # U_mat[i, :] is (m,), V_mat[:, k] is (m,).
                    
                    update_slice = delta * (U_mat[i, :] * V_mat[:, k])
                    
                    old_slice = Rec[i, :, k].copy()
                    Rec[i, :, k] += update_slice
                    
                    new_err = np.max(np.abs(Rec - 1.0))
                    
                    if new_err < best_err - 1e-10:
                        best_err = new_err
                        best_val = cand
                    else:
                        Rec[i, :, k] = old_slice
                
                if best_val != old_val:
                    W[idx, r] = best_val
                    current_err = best_err
                    improved = True
            
            if not improved:
                # If no improvement in this step, maybe try a random perturbation?
                # Or just continue.
                pass

        if current_err < atol:
            return U, V, W, True
        return U, V, W, False

    def search_for_solution(max_rank, max_search_time=140.0):
        nonlocal best_solution, best_rank
        search_start = time.time()
        
        # Try ranks from best_rank - 1 down to a lower bound
        # Lower bound for 3x3x5 is likely around 20-25.
        for rank in range(min(max_rank, best_rank - 1), 15, -1):
            if time.time() > search_start + max_search_time:
                break
            if rank >= best_rank:
                continue
            
            # Try multiple random initializations
            n_attempts = 20 # Increase attempts since local search is faster
            for attempt in range(n_attempts):
                if time.time() > search_start + max_search_time:
                    break
                if time.time() > deadline:
                    return
                
                # Random initialization with small integers/half-integers
                np.random.seed(42 + rank * 1000 + attempt * 7)
                U = np.random.choice([-1, -0.5, 0.5, 1], (n * m, rank))
                V = np.random.choice([-1, -0.5, 0.5, 1], (m * p, rank))
                W = np.random.choice([-1, -0.5, 0.5, 1], (p * n, rank))
                
                # Check if already valid (unlikely)
                if is_valid(U, V, W, rank):
                    best_solution = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    return True
                
                # Try local optimization
                U_opt, V_opt, W_opt, success = local_optimize_fast(U, V, W, rank, max_iters=1000)
                if success:
                    best_solution = {"U": U_opt, "V": V_opt, "W": W_opt}
                    best_rank = rank
                    return True
        
        return False

    # Try to find a better solution
    search_for_solution(best_rank - 1)

    def to_lists(mat):
        return mat.tolist()

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
