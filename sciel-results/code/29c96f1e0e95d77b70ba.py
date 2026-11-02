<<<<<<< SEARCH
            # Phase 1: ALS iterations
            for als_iter in range(80):
                if time.time() - start_time > deadline:
                    return False
                U, V, W = als_step(U, V, W, target_rank, T)
                err = compute_error(U, V, W, target_rank)
                if err < 1e-10:
                    break
            
            # Phase 2: Gradient descent polish with multiple learning rates
            for lr in [0.01, 0.005, 0.001]:
                U, V, W, err = gradient_descent(U, V, W, target_rank, max_iters=150, lr=lr)
                if err is not None and err < 1e-8:
                    break
            
            if err is not None and err < 1e-6:
                # Phase 3: Fine polish with smaller learning rate
                U, V, W, err = gradient_descent(U, V, W, target_rank, max_iters=300, lr=0.0005)
                
                if err is not None and err < 1e-8:
                    # Round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify rounded
                    err_rounded = compute_error(U_rounded, V_rounded, W_rounded, target_rank)
                    if err_rounded < 1e-6:
                        nonlocal best_U, best_V, best_W, best_rank
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        best_rank = target_rank
                        return True
=======
            # Phase 1: ALS iterations
            for als_iter in range(120):
                if time.time() - start_time > deadline:
                    return False
                U, V, W = als_step(U, V, W, target_rank, T)
                err = compute_error(U, V, W, target_rank)
                if err < 1e-10:
                    break
            
            # Phase 2: Gradient descent polish with multiple learning rates
            err = np.inf
            for lr in [0.02, 0.01, 0.005, 0.002]:
                U, V, W, err = gradient_descent(U, V, W, target_rank, max_iters=200, lr=lr)
                if err is not None and err < 1e-8:
                    break
            
            if err is not None and err < 1e-5:
                # Phase 3: Fine polish with smaller learning rate
                U, V, W, err = gradient_descent(U, V, W, target_rank, max_iters=400, lr=0.001)
                
                if err is not None and err < 1e-6:
                    # Try rounding to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify rounded
                    err_rounded = compute_error(U_rounded, V_rounded, W_rounded, target_rank)
                    if err_rounded < 1e-4:
                        nonlocal best_U, best_V, best_W, best_rank
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        best_rank = target_rank
                        return True
                    else:
                        # Try rounding to integers
                        U_rounded_int = np.round(U)
                        V_rounded_int = np.round(V)
                        W_rounded_int = np.round(W)
                        err_rounded_int = compute_error(U_rounded_int, V_rounded_int, W_rounded_int, target_rank)
                        if err_rounded_int < 1e-4:
                            nonlocal best_U, best_V, best_W, best_rank
                            best_U = U_rounded_int
                            best_V = V_rounded_int
                            best_W = W_rounded_int
                            best_rank = target_rank
                            return True
>>>>>>> REPLACE

