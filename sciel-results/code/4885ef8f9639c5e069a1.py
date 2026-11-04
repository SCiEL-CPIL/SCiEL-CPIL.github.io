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

    def local_optimize_fast(U, V, W, rank, max_iters=2000, atol=1e-9):
        """
        Fast local optimization using incremental error updates.
        Expands the candidate set to include more half-integers.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Precompute valid values: half-integers from -5 to 5
        valid_vals = np.array([x / 2.0 for x in range(-10, 11)])
        
        # Initial reconstruction
        Rec = compute_reconstruction(U, V, W, rank)
        current_err = np.max(np.abs(Rec - 1.0))
        
        if current_err < atol:
            return U, V, W, True

        # Helper to get candidates around a value
        def get_candidates(old_val):
            candidates = set()
            # Always include the old value
            candidates.add(old_val)
            # Add neighbors in the valid_vals space
            # Find index of old_val in valid_vals if it exists, else nearest
            # Since old_val might not be in valid_vals initially, we just try a range
            # around it, snapping to valid_vals
            for delta in [-1.0, -0.5, 0.5, 1.0, -2.0, 2.0]:
                c = old_val + delta
                # Snap to nearest 0.5
                c_snapped = np.round(c * 2) / 2
                if c_snapped in valid_vals:
                    candidates.add(c_snapped)
            return list(candidates)

        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            if current_err < atol:
                break
            
            improved = False
            # Randomly select an entry to modify
            mat_type = np.random.randint(3) # 0: U, 1: V, 2: W
            r = np.random.randint(rank)
            
            if mat_type == 0:
                idx = np.random.randint(n * m)
                i = idx // m
                j = idx % m
                old_val = U[idx, r]
                
                best_val = old_val
                best_err = current_err
                
                candidates = get_candidates(old_val)
                
                for cand in candidates:
                    if cand == old_val:
                        continue
                    
                    delta = cand - old_val
                    V_mat = V[:, r].reshape(m, p)
                    W_mat = W[:, r].reshape(p, n)
                    
                    update_slice = delta * (V_mat * W_mat[:, i])
                    
                    old_slice = Rec[i, :, :].copy()
                    Rec[i, :, :] += update_slice
                    
                    new_err = np.max(np.abs(Rec - 1.0))
                    
                    if new_err < best_err - 1e-10:
                        best_err = new_err
                        best_val = cand
                    else:
                        Rec[i, :, :] = old_slice
                
                if best_val != old_val:
                    U[idx, r] = best_val
                    current_err = best_err
                    improved = True
            
            elif mat_type == 1:
                idx = np.random.randint(m * p)
                j = idx // p
                k = idx % p
                old_val = V[idx, r]
                
                best_val = old_val
                best_err = current_err
                
                candidates = get_candidates(old_val)
                
                for cand in candidates:
                    if cand == old_val:
                        continue
                    
                    delta = cand - old_val
                    U_mat = U[:, r].reshape(n, m)
                    W_mat = W[:, r].reshape(p, n)
                    
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
                
                candidates = get_candidates(old_val)
                
                for cand in candidates:
                    if cand == old_val:
                        continue
                    
                    delta = cand - old_val
                    U_mat = U[:, r].reshape(n, m)
                    V_mat = V[:, r].reshape(m, p)
                    
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
                # If stuck for a while, we might try a larger jump, but for now just continue
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
            n_attempts = 30
            for attempt in range(n_attempts):
                if time.time() > search_start + max_search_time:
                    break
                if time.time() > deadline:
                    return
                
                # Strategy 1: Random initialization
                if attempt % 2 == 0:
                    np.random.seed(42 + rank * 1000 + attempt * 7)
                    U = np.random.choice([-1, -0.5, 0, 0.5, 1], (n * m, rank))
                    V = np.random.choice([-1, -0.5, 0, 0.5, 1], (m * p, rank))
                    W = np.random.choice([-1, -0.5, 0, 0.5, 1], (p * n, rank))
                else:
                    # Strategy 2: Structured initialization based on schoolbook
                    # Merge random pairs of schoolbook columns
                    U = np.zeros((n * m, rank))
                    V = np.zeros((m * p, rank))
                    W = np.zeros((p * n, rank))
                    
                    # Pick 'rank' random combinations of (i,j,k) to form the basis
                    # This is a heuristic: we assume the solution is a sparse combination of schoolbook terms
                    # We pick 'rank' random indices from the n*m*p possible terms
                    indices = np.random.choice(n * m * p, rank, replace=False)
                    for r_idx, idx in enumerate(indices):
                        k = idx % p
                        tmp = idx // p
                        j = tmp % m
                        i = tmp // m
                        # Assign random coefficients
                        coeff = np.random.choice([-1, -0.5, 0.5, 1])
                        U[i * m + j, r_idx] = coeff
                        V[j * p + k, r_idx] = 1
                        W[k * n + i, r_idx] = 1
                        # Note: This structured init is very sparse. Local search will need to fill in.
                        # It's better to start with a denser random init for local search to work well.
                        # Let's stick to random but with more diverse values.
                        # Reverting to a better random init for odd attempts
                        U = np.random.choice([-1, -0.5, 0, 0.5, 1], (n * m, rank))
                        V = np.random.choice([-1, -0.5, 0, 0.5, 1], (m * p, rank))
                        W = np.random.choice([-1, -0.5, 0, 0.5, 1], (p * n, rank))

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
