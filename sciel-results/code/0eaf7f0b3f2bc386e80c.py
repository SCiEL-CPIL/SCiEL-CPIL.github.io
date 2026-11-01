<<<<<<< SEARCH
    # Try to find a lower rank decomposition using local search
    # We will try to reduce the rank from 42 to 41, then 40, etc.
    
    def try_rank_search(target_rank, max_iters=500):
        """
        Try to find a decomposition of target_rank.
        Uses a simple local search: start with random/schoolbook mix, perturb, check.
        """
        if time.time() > deadline:
            return None
        
        # Initialize with schoolbook terms, but randomly drop (42 - target_rank) terms
        # and try to fix the remaining ones?
        # Or start with random sparse vectors.
        
        best_error = np.inf
        best_U, best_V, best_W = None, None, None
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.zeros((n * m, target_rank), dtype=np.float64)
            V = np.zeros((m * p, target_rank), dtype=np.float64)
            W = np.zeros((p * n, target_rank), dtype=np.float64)
            
            # Each rank-1 term should be "sparse" to mimic the structure
            for r_idx in range(target_rank):
                # Pick random supports
                # Support size 2-3 is typical for these decompositions
                u_size = np.random.randint(2, 4)
                v_size = np.random.randint(2, 4)
                w_size = np.random.randint(2, 4)
                
                u_pos = np.random.choice(n * m, size=u_size, replace=False)
                v_pos = np.random.choice(m * p, size=v_size, replace=False)
                w_pos = np.random.choice(p * n, size=w_size, replace=False)
                
                # Assign random half-integer values
                U[u_pos, r_idx] = np.random.choice([-2, -1, -0.5, 0.5, 1, 2], size=u_size)
                V[v_pos, r_idx] = np.random.choice([-2, -1, -0.5, 0.5, 1, 2], size=v_size)
                W[w_pos, r_idx] = np.random.choice([-2, -1, -0.5, 0.5, 1, 2], size=w_size)
            
            # Compute error
            T_comp = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            error = np.sum((T_comp - T) ** 2)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if error < 1e-9:
                    # Found a solution!
                    # Round to half-integers to be safe
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    if verify(U_int, V_int, W_int):
                        return U_int, V_int, W_int
            
            # If we found a very good solution but not exact, try to polish it?
            # For now, just keep searching.
            
        return None
=======
    # Try to find a lower rank decomposition using local search
    # We will try to reduce the rank from 42 to 41, then 40, etc.
    
    def compute_error(U, V, W):
        """Compute the squared error of the tensor decomposition."""
        T_comp = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.sum((T_comp - T) ** 2)
    
    def refine_solution(U, V, W, max_iters=200, tol=1e-9):
        """
        Refine a near-miss solution by perturbing individual entries.
        Returns refined solution if it reaches exact match, else None.
        """
        best_error = compute_error(U, V, W)
        if best_error < tol:
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            if verify(U_int, V_int, W_int):
                return U_int, V_int, W_int
            return None
        
        U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Pick a random matrix and entry to perturb
            mat_choice = np.random.choice(3)
            if mat_choice == 0:
                row = np.random.randint(n * m)
                col = np.random.randint(target_rank)
                old_val = U_curr[row, col]
                # Try adding small perturbations
                for delta in [0.5, -0.5, 1.0, -1.0]:
                    new_val = old_val + delta
                    U_curr[row, col] = new_val
                    err = compute_error(U_curr, V_curr, W_curr)
                    if err < best_error:
                        best_error = err
                        if best_error < tol:
                            U_int = np.round(U_curr * 2) / 2
                            V_int = np.round(V_curr * 2) / 2
                            W_int = np.round(W_curr * 2) / 2
                            if verify(U_int, V_int, W_int):
                                return U_int, V_int, W_int
                    else:
                        U_curr[row, col] = old_val
                        break
            elif mat_choice == 1:
                row = np.random.randint(m * p)
                col = np.random.randint(target_rank)
                old_val = V_curr[row, col]
                for delta in [0.5, -0.5, 1.0, -1.0]:
                    new_val = old_val + delta
                    V_curr[row, col] = new_val
                    err = compute_error(U_curr, V_curr, W_curr)
                    if err < best_error:
                        best_error = err
                        if best_error < tol:
                            U_int = np.round(U_curr * 2) / 2
                            V_int = np.round(V_curr * 2) / 2
                            W_int = np.round(W_curr * 2) / 2
                            if verify(U_int, V_int, W_int):
                                return U_int, V_int, W_int
                    else:
                        V_curr[row, col] = old_val
                        break
            else:
                row = np.random.randint(p * n)
                col = np.random.randint(target_rank)
                old_val = W_curr[row, col]
                for delta in [0.5, -0.5, 1.0, -1.0]:
                    new_val = old_val + delta
                    W_curr[row, col] = new_val
                    err = compute_error(U_curr, V_curr, W_curr)
                    if err < best_error:
                        best_error = err
                        if best_error < tol:
                            U_int = np.round(U_curr * 2) / 2
                            V_int = np.round(V_curr * 2) / 2
                            W_int = np.round(W_curr * 2) / 2
                            if verify(U_int, V_int, W_int):
                                return U_int, V_int, W_int
                    else:
                        W_curr[row, col] = old_val
                        break
        
        return None
    
    def try_rank_search(target_rank, max_iters=300):
        """
        Try to find a decomposition of target_rank.
        Uses smart initialization based on schoolbook structure plus local refinement.
        """
        if time.time() > deadline:
            return None
        
        best_error = np.inf
        best_U, best_V, best_W = None, None, None
        
        # Strategy 1: Start with schoolbook, drop random terms, and refine
        for attempt in range(min(50, max_iters)):
            if time.time() > deadline:
                break
            
            # Start with schoolbook and randomly zero out (42 - target_rank) columns
            U = U_sb.copy()
            V = V_sb.copy()
            W = W_sb.copy()
            
            # Select columns to keep
            cols_to_keep = np.random.choice(rank_schoolbook, size=target_rank, replace=False)
            cols_to_keep = np.sort(cols_to_keep)
            
            U = U[:, cols_to_keep]
            V = V[:, cols_to_keep]
            W = W[:, cols_to_keep]
            
            # Try to adjust values to compensate for dropped terms
            # First, compute current error
            error = compute_error(U, V, W)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if error < 1e-9:
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    if verify(U_int, V_int, W_int):
                        return U_int, V_int, W_int
            
            # Try to refine this solution
            refined = refine_solution(U, V, W, max_iters=50)
            if refined is not None:
                return refined
            
            # If refinement didn't work, try perturbing the kept terms
            # Randomly adjust some entries
            for _ in range(10):
                if time.time() > deadline:
                    break
                    
                # Pick a random entry to perturb
                mat_choice = np.random.choice(3)
                if mat_choice == 0:
                    row = np.random.randint(n * m)
                    col = np.random.randint(target_rank)
                    U[row, col] += np.random.choice([-1, -0.5, 0.5, 1])
                elif mat_choice == 1:
                    row = np.random.randint(m * p)
                    col = np.random.randint(target_rank)
                    V[row, col] += np.random.choice([-1, -0.5, 0.5, 1])
                else:
                    row = np.random.randint(p * n)
                    col = np.random.randint(target_rank)
                    W[row, col] += np.random.choice([-1, -0.5, 0.5, 1])
                
                error = compute_error(U, V, W)
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                    if error < 1e-9:
                        U_int = np.round(U * 2) / 2
                        V_int = np.round(V * 2) / 2
                        W_int = np.round(W * 2) / 2
                        if verify(U_int, V_int, W_int):
                            return U_int, V_int, W_int
            
            # Try refinement on the perturbed solution
            if best_error < 10.0:  # Only refine if we're getting close
                refined = refine_solution(best_U, best_V, best_W, max_iters=30)
                if refined is not None:
                    return refined
        
        # Strategy 2: Try random sparse initialization with refinement
        remaining_iters = max_iters - min(50, max_iters)
        for _ in range(remaining_iters):
            if time.time() > deadline:
                break
                
            # Random sparse initialization
            U = np.zeros((n * m, target_rank), dtype=np.float64)
            V = np.zeros((m * p, target_rank), dtype=np.float64)
            W = np.zeros((p * n, target_rank), dtype=np.float64)
            
            for r_idx in range(target_rank):
                u_size = np.random.randint(2, 4)
                v_size = np.random.randint(2, 4)
                w_size = np.random.randint(2, 4)
                
                u_pos = np.random.choice(n * m, size=u_size, replace=False)
                v_pos = np.random.choice(m * p, size=v_size, replace=False)
                w_pos = np.random.choice(p * n, size=w_size, replace=False)
                
                U[u_pos, r_idx] = np.random.choice([-2, -1, -0.5, 0.5, 1, 2], size=u_size)
                V[v_pos, r_idx] = np.random.choice([-2, -1, -0.5, 0.5, 1, 2], size=v_size)
                W[w_pos, r_idx] = np.random.choice([-2, -1, -0.5, 0.5, 1, 2], size=w_size)
            
            error = compute_error(U, V, W)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if error < 1e-9:
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    if verify(U_int, V_int, W_int):
                        return U_int, V_int, W_int
            
            # Try refinement if error is reasonably low
            if error < 20.0:
                refined = refine_solution(U, V, W, max_iters=20)
                if refined is not None:
                    return refined
        
        return None
>>>>>>> REPLACE

