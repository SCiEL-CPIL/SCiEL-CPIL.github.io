<<<<<<< SEARCH
def _try_rank_optimization(target_rank, n, m, p, deadline, max_attempts=50):
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T_exp = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
                
    U0, V0, W0 = _schoolbook(n, m, p)
    rng = random.Random(42)
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        # Initialization Strategy:
        # 1. Pick target_rank columns from schoolbook
        # 2. Add small noise
        # 3. Or random init
        
        if attempt < 20:
            # Schoolbook subset
            cols = rng.sample(range(27), target_rank)
            U_init = U0[:, cols].copy()
            V_init = V0[:, cols].copy()
            W_init = W0[:, cols].copy()
            # Perturb to escape local minima
            U_init += np.random.randn(dim_a, target_rank) * 0.01
            V_init += np.random.randn(dim_b, target_rank) * 0.01
            W_init += np.random.randn(dim_c, target_rank) * 0.01
        else:
            # Random init
            scale = 0.5
            U_init = np.random.randn(dim_a, target_rank) * scale
            V_init = np.random.randn(dim_b, target_rank) * scale
            W_init = np.random.randn(dim_c, target_rank) * scale

        U, V, W = U_init, V_init, W_init
        
        # Run ALS
        for it in range(200):
            if time.time() > deadline:
                break
            U, V, W = _als_step(U, V, W, T_exp, n, m, p, target_rank)
            
            # Check error
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.sum((T_approx - T_exp)**2)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if error < 1e-10:
                return U, V, W
                
    if best_error < 1e-7:
        return best_U, best_V, best_W
        
    return None
=======
def _try_rank_optimization(target_rank, n, m, p, deadline, max_attempts=50):
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T_exp = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
                
    U0, V0, W0 = _schoolbook(n, m, p)
    rng = random.Random(42)
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        # Initialization Strategy:
        # 1. Pick target_rank columns from schoolbook
        # 2. Add small noise
        # 3. Or random init
        
        if attempt < 30:
            # Schoolbook subset with varying noise levels
            cols = rng.sample(range(27), target_rank)
            U_init = U0[:, cols].copy()
            V_init = V0[:, cols].copy()
            W_init = W0[:, cols].copy()
            # Perturb to escape local minima with varying noise
            noise_scale = 0.01 * (1 + attempt * 0.1)
            U_init += np.random.randn(dim_a, target_rank) * noise_scale
            V_init += np.random.randn(dim_b, target_rank) * noise_scale
            W_init += np.random.randn(dim_c, target_rank) * noise_scale
        elif attempt < 40:
            # Random init with moderate scale
            scale = 0.3 + 0.02 * (attempt - 30)
            U_init = np.random.randn(dim_a, target_rank) * scale
            V_init = np.random.randn(dim_b, target_rank) * scale
            W_init = np.random.randn(dim_c, target_rank) * scale
        else:
            # Sparse random init
            U_init = np.random.randn(dim_a, target_rank) * 0.5
            V_init = np.random.randn(dim_b, target_rank) * 0.5
            W_init = np.random.randn(dim_c, target_rank) * 0.5
            # Zero out some entries to create sparsity
            mask_U = np.random.rand(dim_a, target_rank) < 0.3
            mask_V = np.random.rand(dim_b, target_rank) < 0.3
            mask_W = np.random.rand(dim_c, target_rank) < 0.3
            U_init[mask_U] = 0
            V_init[mask_V] = 0
            W_init[mask_W] = 0

        U, V, W = U_init, V_init, W_init
        
        # Run hybrid optimization
        U, V, W = _hybrid_optimization(U, V, W, T_exp, n, m, p, target_rank, deadline, 
                                       max_als_iters=50, max_gd_iters=30)
            
        # Check error
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.sum((T_approx - T_exp)**2)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if error < 1e-10:
            return U, V, W
            
    if best_error < 1e-7:
        return best_U, best_V, best_W
        
    return None
>>>>>>> REPLACE

