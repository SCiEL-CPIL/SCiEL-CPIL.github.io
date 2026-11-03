    def try_optimize_block(rank_block, max_time=20):
        """Try to find a rank_block decomposition for 3x3x2 using optimization."""
        if time_left() < max_time + 5:
            return None
        
        n_params = 21 * rank_block
        
        def unpack(params):
            U = params[0:9*rank_block].reshape(9, rank_block)
            V = params[9*rank_block:15*rank_block].reshape(6, rank_block)
            W = params[15*rank_block:21*rank_block].reshape(6, rank_block)
            return U, V, W
        
        def objective(params):
            U, V, W = unpack(params)
            T_computed = np.zeros((3, 3, 2))
            for c in range(rank_block):
                u_mat = U[:, c].reshape(3, 3)
                v_mat = V[:, c].reshape(3, 2)
                w_mat = W[:, c].reshape(2, 3)
                for i in range(3):
                    for j in range(3):
                        if abs(u_mat[i,j]) < 1e-15:
                            continue
                        for k in range(2):
                            T_computed[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            return np.sum((T_computed - 1.0)**2)
        
        best_params = None
        best_obj = float('inf')
        
        # Use a mix of random initializations
        # 1. Truncated rank-17 (current)
        # 2. Random small integers/half-integers
        # 3. Random Gaussian
        
        U17, V17, W17 = get_17_rank_block()
        
        n_restarts = 10
        for restart in range(n_restarts):
            if time_left() < max_time:
                break
            
            if restart == 0:
                # Truncated
                x0 = np.zeros(n_params)
                if rank_block <= 17:
                    x0[0:9*rank_block] = U17[:, :rank_block].flatten()
                    x0[9*rank_block:15*rank_block] = V17[:, :rank_block].flatten()
                    x0[15*rank_block:21*rank_block] = W17[:, :rank_block].flatten()
            elif restart < 5:
                # Random half-integers in [-2, 2]
                x0 = np.round(4 * np.random.rand(n_params) - 2, 1)
            else:
                # Random Gaussian
                x0 = np.random.randn(n_params) * 0.5
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                 options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x
            except Exception:
                pass
        
        if best_params is None or best_obj > 1e-8:
            return None
        
        U, V, W = unpack(best_params)
        
        # Verify
        T_check = np.zeros((3, 3, 2))
        for c in range(rank_block):
            u_mat = U[:, c].reshape(3, 3)
            v_mat = V[:, c].reshape(3, 2)
            w_mat = W[:, c].reshape(2, 3)
            for i in range(3):
                for j in range(3):
                    for k in range(2):
                        T_check[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        
        if np.max(np.abs(T_check - 1.0)) > 1e-8:
            return None
        
        return U, V, W
