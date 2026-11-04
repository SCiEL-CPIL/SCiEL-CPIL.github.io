    def try_find_solution_warm(R, num_trials=10):
        # Use rank 28 solution as base
        # Construct rank 28 solution first
        U28, V28, W28 = construct_rank_28()
        
        # Remove 28-R columns
        # We need to choose which columns to remove.
        # Randomly select columns to keep?
        # Or remove the last ones?
        # Let's try removing random columns.
        
        for t in range(num_trials):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Select R columns to keep
            cols_to_keep = np.random.choice(28, R, replace=False)
            cols_to_keep.sort()
            
            U_init = U28[:, cols_to_keep]
            V_init = V28[:, cols_to_keep]
            W_init = W28[:, cols_to_keep]
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def residuals(x):
                U = x[:8*R].reshape(8, R)
                V = x[8*R:8*R+16*R].reshape(16, R)
                W = x[8*R+16*R:].reshape(8, R)
                T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
                return T_est.flatten() - T_flat
            
            try:
                res = least_squares(residuals, x0, method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if res.cost < 1e-10:
                U_sol = res.x[:8*R].reshape(8, R)
                V_sol = res.x[8*R:8*R+16*R].reshape(16, R)
                W_sol = res.x[8*R+16*R:].reshape(8, R)
                
                U_round = np.round(U_sol * 2) / 2
                V_round = np.round(V_sol * 2) / 2
                W_round = np.round(W_sol * 2) / 2
                
                if verify_solution(U_round, V_round, W_round, R):
                    return U_round.tolist(), V_round.tolist(), W_round.tolist()
                
                U_int = np.round(U_sol)
                V_int = np.round(V_sol)
                W_int = np.round(W_sol)
                if verify_solution(U_int, V_int, W_int, R):
                    return U_int.tolist(), V_int.tolist(), W_int.tolist()
            
        return None
