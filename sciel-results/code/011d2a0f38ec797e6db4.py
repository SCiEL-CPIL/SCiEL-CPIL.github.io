<<<<<<< SEARCH
def _try_rank(n, m, p, rank, deadline):
    num_params = (n * m + m * p + p * n) * rank
    u_size = n * m * rank
    v_size = m * p * rank
    w_size = p * n * rank
    
    # Precompute delta
    delta = np.eye(m).reshape(1, 1, m, m) # shape (1, 1, m, m)
    
    def residual_function(params):
        U = params[:u_size].reshape(n, m, rank)
        V = params[u_size:u_size + v_size].reshape(m, p, rank)
        W = params[u_size + v_size:].reshape(p, n, rank)
        
        # Compute sum_r U[i,j,r] V[j2,k,r] W[k,i,r]
        # U: (n, m, r)
        # V: (m, p, r) -> indices j2, k, r
        # W: (p, n, r) -> indices k, i, r
        # Result: (n, p, m, m)
        # einsum: 'ijr,j2kr,kinr->ikjj2'
        try:
            prod = np.einsum('ijr,j2kr,kinr->ikjj2', U, V, W)
        except Exception:
            # Fallback to loops if einsum fails (unlikely)
            prod = np.zeros((n, p, m, m))
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            val = 0.0
                            for r in range(rank):
                                val += U[i, j1, r] * V[j2, k, r] * W[k, i, r]
                            prod[i, k, j1, j2] = val
                            
        residual = prod - delta
        return residual.flatten()

    # Initializations
    inits = []
    
    # 1. Random
    for _ in range(3):
        np.random.seed(int(time.time() * 10000) % 100000)
        U0 = np.random.randn(n, m, rank)
        V0 = np.random.randn(m, p, rank)
        W0 = np.random.randn(p, n, rank)
        scale = 1.0 / np.sqrt(rank)
        U0 *= scale
        V0 *= scale
        W0 *= scale
        inits.append(np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()]))
        
    # 2. Perturbed Schoolbook (if rank is close to schoolbook)
    # This is tricky because schoolbook has rank 24.
    # We can take the first 'rank' columns of schoolbook and randomize the rest?
    # Or just use random.
    
    for init in inits:
        if time.time() > deadline:
            break
        
        try:
            result = least_squares(residual_function, init, method='trf', max_nfev=200)
        except Exception:
            continue
            
        if result.cost < 1e-8:
            U_opt = result.x[:u_size].reshape(n, m, rank)
            V_opt = result.x[u_size:u_size + v_size].reshape(m, p, rank)
            W_opt = result.x[u_size + v_size:].reshape(p, n, rank)
            
            # Snap to half-integers
            for denom in [1, 2, 3, 4]:
                U_snap = np.round(U_opt * denom) / denom
                V_snap = np.round(V_opt * denom) / denom
                W_snap = np.round(W_opt * denom) / denom
                
                # Verify
                max_err = 0.0
                try:
                    prod = np.einsum('ijr,j2kr,kinr->ikjj2', U_snap, V_snap, W_snap)
                    err = np.max(np.abs(prod - delta))
                except Exception:
                    err = 1.0
                    
                if err < 1e-9:
                    # Convert back to list of lists
                    U_flat = U_snap.flatten().tolist()
                    V_flat = V_snap.flatten().tolist()
                    W_flat = W_snap.flatten().tolist()
                    
                    # Reshape to (rows, rank)
                    U_out = [U_flat[i*rank:(i+1)*rank] for i in range(n*m)]
                    V_out = [V_flat[i*rank:(i+1)*rank] for i in range(m*p)]
                    W_out = [W_flat[i*rank:(i+1)*rank] for i in range(p*n)]
                    
                    return {"U": U_out, "V": V_out, "W": W_out}
                    
    return None
=======
def _try_rank(n, m, p, rank, deadline):
    num_params = (n * m + m * p + p * n) * rank
    u_size = n * m * rank
    v_size = m * p * rank
    w_size = p * n * rank
    
    # Precompute delta
    delta = np.eye(m).reshape(1, 1, m, m) # shape (1, 1, m, m)
    
    def residual_function(params):
        U = params[:u_size].reshape(n, m, rank)
        V = params[u_size:u_size + v_size].reshape(m, p, rank)
        W = params[u_size + v_size:].reshape(p, n, rank)
        
        # Compute sum_r U[i,j,r] V[j2,k,r] W[k,i,r]
        # U: (n, m, r)
        # V: (m, p, r) -> indices j2, k, r
        # W: (p, n, r) -> indices k, i, r
        # Result: (n, p, m, m)
        # einsum: 'ijr,j2kr,kinr->ikjj2'
        try:
            prod = np.einsum('ijr,j2kr,kinr->ikjj2', U, V, W)
        except Exception:
            # Fallback to loops if einsum fails (unlikely)
            prod = np.zeros((n, p, m, m))
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            val = 0.0
                            for r in range(rank):
                                val += U[i, j1, r] * V[j2, k, r] * W[k, i, r]
                            prod[i, k, j1, j2] = val
                            
        residual = prod - delta
        return residual.flatten()

    def _verify_and_snap(U_opt, V_opt, W_opt):
        """Try to snap to half-integers and verify."""
        for denom in [1, 2, 3, 4, 6, 8]:
            U_snap = np.round(U_opt * denom) / denom
            V_snap = np.round(V_opt * denom) / denom
            W_snap = np.round(W_opt * denom) / denom
            
            # Verify
            try:
                prod = np.einsum('ijr,j2kr,kinr->ikjj2', U_snap, V_snap, W_snap)
                err = np.max(np.abs(prod - delta))
            except Exception:
                err = 1.0
                
            if err < 1e-9:
                # Convert back to list of lists
                U_flat = U_snap.flatten().tolist()
                V_flat = V_snap.flatten().tolist()
                W_flat = W_snap.flatten().tolist()
                
                # Reshape to (rows, rank)
                U_out = [U_flat[i*rank:(i+1)*rank] for i in range(n*m)]
                V_out = [V_flat[i*rank:(i+1)*rank] for i in range(m*p)]
                W_out = [W_flat[i*rank:(i+1)*rank] for i in range(p*n)]
                
                return {"U": U_out, "V": V_out, "W": W_out}
        return None

    # Geometric reparametrization: apply random linear transformations
    def _generate_reparam_init(seed_val):
        """Generate initial condition with random linear reparametrization."""
        np.random.seed(seed_val)
        
        # Generate random invertible matrices for each factor space
        # These represent changes of basis in the factor spaces
        A_u = np.random.randn(n * m, n * m) + 10 * np.eye(n * m)
        A_v = np.random.randn(m * p, m * p) + 10 * np.eye(m * p)
        A_w = np.random.randn(p * n, p * n) + 10 * np.eye(p * n)
        
        # Start with a base random decomposition
        U_base = np.random.randn(n, m, rank) * (1.0 / np.sqrt(rank))
        V_base = np.random.randn(m, p, rank) * (1.0 / np.sqrt(rank))
        W_base = np.random.randn(p, n, rank) * (1.0 / np.sqrt(rank))
        
        # Apply reparametrization: U_new = U_base @ A_u^{-1}, etc.
        # This changes the coordinate system but preserves the tensor
        try:
            A_u_inv = np.linalg.inv(A_u)
            A_v_inv = np.linalg.inv(A_v)
            A_w_inv = np.linalg.inv(A_w)
            
            U_flat = U_base.reshape(n * m, rank)
            V_flat = V_base.reshape(m * p, rank)
            W_flat = W_base.reshape(p * n, rank)
            
            U_new = (U_flat @ A_u_inv.T).reshape(n, m, rank)
            V_new = (V_flat @ A_v_inv.T).reshape(m, p, rank)
            W_new = (W_flat @ A_w_inv.T).reshape(p, n, rank)
        except np.linalg.LinAlgError:
            U_new = U_base
            V_new = V_base
            W_new = W_base
        
        return np.concatenate([U_new.flatten(), V_new.flatten(), W_new.flatten()])

    # Generate multiple initializations with different reparametrizations
    inits = []
    base_seed = int(time.time() * 10000) % 100000
    
    # Plain random inits
    for i in range(2):
        np.random.seed(base_seed + i)
        U0 = np.random.randn(n, m, rank) / np.sqrt(rank)
        V0 = np.random.randn(m, p, rank) / np.sqrt(rank)
        W0 = np.random.randn(p, n, rank) / np.sqrt(rank)
        inits.append(np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()]))
    
    # Reparametrized inits to escape local minima
    for i in range(4):
        init = _generate_reparam_init(base_seed + 100 + i)
        inits.append(init)
    
    # Try each initialization
    for init in inits:
        if time.time() > deadline:
            break
        
        try:
            result = least_squares(residual_function, init, method='trf', max_nfev=300,
                                   xtol=1e-10, ftol=1e-10, gtol=1e-10)
        except Exception:
            continue
            
        if result.cost < 1e-8:
            U_opt = result.x[:u_size].reshape(n, m, rank)
            V_opt = result.x[u_size:u_size + v_size].reshape(m, p, rank)
            W_opt = result.x[u_size + v_size:].reshape(p, n, rank)
            
            # First try to snap
            snapped = _verify_and_snap(U_opt, V_opt, W_opt)
            if snapped is not None:
                return snapped
            
            # If snap failed, do a high-precision polish pass
            if time.time() < deadline - 10:
                try:
                    result2 = least_squares(residual_function, result.x, method='trf', 
                                           max_nfev=500, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                    if result2.cost < 1e-10:
                        U_opt2 = result2.x[:u_size].reshape(n, m, rank)
                        V_opt2 = result2.x[u_size:u_size + v_size].reshape(m, p, rank)
                        W_opt2 = result2.x[u_size + v_size:].reshape(p, n, rank)
                        
                        snapped2 = _verify_and_snap(U_opt2, V_opt2, W_opt2)
                        if snapped2 is not None:
                            return snapped2
                except Exception:
                    pass
                    
    return None
>>>>>>> REPLACE
