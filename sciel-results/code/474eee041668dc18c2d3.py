    def try_local_search(R, time_budget=20):
        """Try to find a rank-R decomposition via local search from rank-39."""
        t_start = time.time()
        
        U39 = np.array(fallback_result['U'], dtype=float)
        V39 = np.array(fallback_result['V'], dtype=float)
        W39 = np.array(fallback_result['W'], dtype=float)
        
        np.random.seed(42)
        best_err = np.inf
        best_solution = None
        candidates = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        for attempt in range(10):
            if time.time() - t_start > time_budget - 2:
                break
            
            # Initialize: take first R columns of rank-39 solution
            U = U39[:, :R].copy()
            V = V39[:, :R].copy()
            W = W39[:, :R].copy()
            
            # Randomly perturb some entries to escape local minima
            for _ in range(50):
                if time.time() - t_start > time_budget - 1:
                    break
                choice = np.random.random()
                if choice < 0.2:
                    row = np.random.randint(num_rows_U)
                    mat = U
                elif choice < 0.6:
                    row = np.random.randint(num_rows_V)
                    mat = V
                else:
                    row = np.random.randint(num_rows_W)
                    mat = W
                col = np.random.randint(R)
                mat[row, col] = np.random.choice(candidates)
            
            err = compute_error(U, V, W)
            if err < 1e-9:
                if check_decomposition(U.tolist(), V.tolist(), W.tolist()):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Greedy Coordinate Descent (GCD)
            max_sweeps = 20
            for sweep in range(max_sweeps):
                if time.time() - t_start > time_budget - 1:
                    break
                
                improved = False
                
                # U entries
                for i in range(num_rows_U):
                    for r in range(R):
                        old_val = U[i, r]
                        best_val = old_val
                        best_err_local = err
                        for c_val in candidates:
                            if c_val == old_val:
                                continue
                            U[i, r] = c_val
                            e = compute_error(U, V, W)
                            if e < best_err_local:
                                best_err_local = e
                                best_val = c_val
                        U[i, r] = best_val
                        if best_val != old_val:
                            err = best_err_local
                            improved = True
                
                # V entries
                for i in range(num_rows_V):
                    for r in range(R):
                        old_val = V[i, r]
                        best_val = old_val
                        best_err_local = err
                        for c_val in candidates:
                            if c_val == old_val:
                                continue
                            V[i, r] = c_val
                            e = compute_error(U, V, W)
                            if e < best_err_local:
                                best_err_local = e
                                best_val = c_val
                        V[i, r] = best_val
                        if best_val != old_val:
                            err = best_err_local
                            improved = True
                
                # W entries
                for i in range(num_rows_W):
                    for r in range(R):
                        old_val = W[i, r]
                        best_val = old_val
                        best_err_local = err
                        for c_val in candidates:
                            if c_val == old_val:
                                continue
                            W[i, r] = c_val
                            e = compute_error(U, V, W)
                            if e < best_err_local:
                                best_err_local = e
                                best_val = c_val
                        W[i, r] = best_val
                        if best_val != old_val:
                            err = best_err_local
                            improved = True
                
                if err < 1e-9:
                    if check_decomposition(U.tolist(), V.tolist(), W.tolist()):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
                
                if not improved:
                    break
                
                if err < best_err:
                    best_err = err
                    best_solution = (U.copy(), V.copy(), W.copy())
        
        # Final polish on best solution found
        if best_solution is not None and best_err < 1e-4:
            U, V, W = best_solution
            # High-precision polish with GCD
            for _ in range(50):
                if time.time() - t_start > time_budget - 0.5:
                    break
                err = compute_error(U, V, W)
                if err < 1e-12:
                    break
                improved = False
                for i in range(num_rows_U):
                    for r in range(R):
                        old_val = U[i, r]
                        best_val = old_val
                        best_err_local = err
                        for c_val in candidates:
                            if c_val == old_val:
                                continue
                            U[i, r] = c_val
                            e = compute_error(U, V, W)
                            if e < best_err_local:
                                best_err_local = e
                                best_val = c_val
                        U[i, r] = best_val
                        if best_val != old_val:
                            err = best_err_local
                            improved = True
                for i in range(num_rows_V):
                    for r in range(R):
                        old_val = V[i, r]
                        best_val = old_val
                        best_err_local = err
                        for c_val in candidates:
                            if c_val == old_val:
                                continue
                            V[i, r] = c_val
                            e = compute_error(U, V, W)
                            if e < best_err_local:
                                best_err_local = e
                                best_val = c_val
                        V[i, r] = best_val
                        if best_val != old_val:
                            err = best_err_local
                            improved = True
                for i in range(num_rows_W):
                    for r in range(R):
                        old_val = W[i, r]
                        best_val = old_val
                        best_err_local = err
                        for c_val in candidates:
                            if c_val == old_val:
                                continue
                            W[i, r] = c_val
                            e = compute_error(U, V, W)
                            if e < best_err_local:
                                best_err_local = e
                                best_val = c_val
                        W[i, r] = best_val
                        if best_val != old_val:
                            err = best_err_local
                            improved = True
                if not improved:
                    break
            if check_decomposition(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
