# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W, rank
    
    # Check if a decomposition is valid
    def check_valid(U, V, W, tol=1e-10):
        r = U.shape[1]
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        T_target = np.ones((n, m, p))
        return np.allclose(T_rec, T_target, atol=tol)
        
    def snap_to_half_int(A):
        return np.round(A * 2) / 2.0

    # Try to find a lower rank decomposition using optimization
    def try_optimize(rank, seed=None):
        """Try to find a valid decomposition of given rank using optimization."""
        if time.time() > deadline:
            return None
            
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(int(time.time() * 1000) % 100000)

        # Initialize with small random values to help convergence
        scale = 0.1
        U = np.random.randn(n * m, rank) * scale
        V = np.random.randn(m * p, rank) * scale
        W = np.random.randn(p * n, rank) * scale
        
        # Objective: minimize ||T_rec - T_target||^2 + lambda * ||2X - round(2X)||^2
        lambda_penalty = 1.0
        
        def objective(params):
            U_flat, V_flat, W_flat = params
            U_3d = U_flat.reshape(n, m, rank)
            V_3d = V_flat.reshape(m, p, rank)
            W_3d = W_flat.reshape(p, n, rank)
            T_rec = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            T_target = np.ones((n, m, p))
            err = T_rec - T_target
            data_err = np.sum(err ** 2)
            
            # Penalty for deviating from half-integers
            pen_U = np.sum((2 * U_3d - np.round(2 * U_3d)) ** 2)
            pen_V = np.sum((2 * V_3d - np.round(2 * V_3d)) ** 2)
            pen_W = np.sum((2 * W_3d - np.round(2 * W_3d)) ** 2)
            
            return data_err + lambda_penalty * (pen_U + pen_V + pen_W)
        
        # Gradient
        def gradient(params):
            U_flat, V_flat, W_flat = params
            U_3d = U_flat.reshape(n, m, rank)
            V_3d = V_flat.reshape(m, p, rank)
            W_3d = W_flat.reshape(p, n, rank)
            T_rec = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            T_target = np.ones((n, m, p))
            err = T_rec - T_target
            
            # Data error gradient
            dU_data = 2 * np.einsum('ijk,jkc->ijc', err, W_3d)
            dV_data = 2 * np.einsum('ijk,kic->jkc', err, U_3d)
            dW_data = 2 * np.einsum('ijk,jkc->kic', err, V_3d)
            
            # Penalty gradient (subgradient)
            dU_pen = lambda_penalty * 4 * (2 * U_3d - np.round(2 * U_3d))
            dV_pen = lambda_penalty * 4 * (2 * V_3d - np.round(2 * V_3d))
            dW_pen = lambda_penalty * 4 * (2 * W_3d - np.round(2 * W_3d))
            
            dU = dU_data + dU_pen
            dV = dV_data + dV_pen
            dW = dW_data + dW_pen
            
            return [dU.flatten(), dV.flatten(), dW.flatten()]
        
        # Flatten parameters
        def pack(U, V, W):
            return [U.flatten(), V.flatten(), W.flatten()]
        
        def unpack(params):
            U_flat, V_flat, W_flat = params
            return U_flat.reshape(n, m, rank), V_flat.reshape(m, p, rank), W_flat.reshape(p, n, rank)
        
        # Use L-BFGS-B optimization
        try:
            result = opt.minimize(
                objective,
                pack(U, V, W),
                method='L-BFGS-B',
                jac=gradient,
                options={'maxiter': 30, 'maxfun': 300, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            # Check if optimization converged to a low error
            if result.fun < 1e-4:
                U_opt, V_opt, W_opt = unpack(result.x)
                
                # Snap to half-integers
                U_snap = snap_to_half_int(U_opt)
                V_snap = snap_to_half_int(V_opt)
                W_snap = snap_to_half_int(W_opt)
                
                # Verify snapped version
                if check_valid(U_snap, V_snap, W_snap, tol=1e-6):
                    return U_snap.reshape(n*m, rank), V_snap.reshape(m*p, rank), W_snap.reshape(p*n, rank), rank
        except Exception:
            pass
        
        return None
    
    # Try various ranks from low to high
    min_possible = max(n * m, m * p, p * n)
    
    best_result = None
    best_rank = n * m * p
    
    ranks_to_try = list(range(min_possible, 71))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        seeds = [42, 123, 999, 777]
        
        found = False
        for seed in seeds:
            if time.time() > deadline - 5:
                break
                
            result = try_optimize(rank, seed=seed)
            if result is not None:
                U, V, W, r = result
                best_result = (U, V, W, r)
                best_rank = r
                found = True
                break
        
        if found:
            break
    
    if best_result is None:
        U, V, W, rank = schoolbook()
    else:
        U, V, W, rank = best_result
    
    if not check_valid(U, V, W, tol=1e-8):
        U, V, W, rank = schoolbook()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
