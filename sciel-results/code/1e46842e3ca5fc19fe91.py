<<<<<<< SEARCH
    # Try Rank 25 with warm start from rank-26 (drop one column)
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        for R in [25, 24]:
            if time.time() > deadline - 10:
                break
            
            found = False
            for drop_col in range(min(26, R+1)):
                if time.time() > deadline - 10:
                    break
                
                U_init = np.delete(U26, drop_col, axis=1)
                V_init = np.delete(V26, drop_col, axis=1)
                W_init = np.delete(W26, drop_col, axis=1)
                
                if U_init.shape[1] < R:
                    pad_size = R - U_init.shape[1]
                    U_init = np.hstack([U_init, np.random.randn(nm, pad_size) * 0.1])
                    V_init = np.hstack([V_init, np.random.randn(mp, pad_size) * 0.1])
                    W_init = np.hstack([W_init, np.random.randn(pn, pad_size) * 0.1])
                
                res = try_optimize_rank(R, time_budget=5.0, init_U=U_init, init_V=V_init, init_W=W_init)
                if res is not None:
                    best_U, best_V, best_W = res
                    best_rank = R
                    found = True
                    break
            
            if found:
                break
=======
    # Try Rank 25 with warm start from rank-26 (drop one column)
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        for R in [25, 24]:
            if time.time() > deadline - 10:
                break
            
            found = False
            # Try dropping different columns and also random perturbations
            for drop_col in range(min(26, R+1)):
                if time.time() > deadline - 10:
                    break
                
                U_init = np.delete(U26, drop_col, axis=1)
                V_init = np.delete(V26, drop_col, axis=1)
                W_init = np.delete(W26, drop_col, axis=1)
                
                if U_init.shape[1] < R:
                    pad_size = R - U_init.shape[1]
                    U_init = np.hstack([U_init, np.random.randn(nm, pad_size) * 0.1])
                    V_init = np.hstack([V_init, np.random.randn(mp, pad_size) * 0.1])
                    W_init = np.hstack([W_init, np.random.randn(pn, pad_size) * 0.1])
                
                # Add small random perturbation to help escape local minima
                U_init = U_init + np.random.randn(nm, R) * 0.05
                V_init = V_init + np.random.randn(mp, R) * 0.05
                W_init = W_init + np.random.randn(pn, R) * 0.05
                
                res = try_optimize_rank(R, time_budget=4.0, init_U=U_init, init_V=V_init, init_W=W_init)
                if res is not None:
                    best_U, best_V, best_W = res
                    best_rank = R
                    found = True
                    break
            
            if found:
                break
>>>>>>> REPLACE
