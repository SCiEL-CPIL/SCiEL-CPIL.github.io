<<<<<<< SEARCH
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 40s for safety and fallback
    
    rank_try = 20
    best_result = None
    best_error = np.inf
    
    # Try multiple random initializations
    num_attempts = 15
    for attempt in range(num_attempts):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(100 + attempt)
        # Scale initial values appropriately
        x0 = np.random.randn(nm * rank_try + mp * rank_try + pn * rank_try) * 1.0
        
        try:
            result = minimize(
                objective, 
                x0, 
                args=(rank_try,),
                method='L-BFGS-B', 
                jac=gradient,
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < best_error:
                best_error = result.fun
                best_result = result.x
                
                # If we found a very good solution, do a polish pass
                if best_error < 1e-4:
                    result2 = minimize(
                        objective, 
                        best_result, 
                        args=(rank_try,),
                        method='L-BFGS-B', 
                        jac=gradient,
                        options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if result2.fun < best_error:
                        best_error = result2.fun
                        best_result = result2.x
                        
        except Exception as e:
            pass
=======
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 35s for safety and fallback
    
    rank_try = 20
    best_result = None
    best_error = np.inf
    
    # Generate structured initializations based on Strassen-like patterns
    def generate_structured_init(seed):
        """Generate initializations with structure that may help find rank-20 solutions."""
        np.random.seed(seed)
        U = np.zeros((nm, rank_try))
        V = np.zeros((mp, rank_try))
        W = np.zeros((pn, rank_try))
        
        # Start from a rank-21 Strassen-like structure and perturb
        # First 14 ranks: two blocks of Strassen (7 each)
        strassen_steps = [
            {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
            {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
            {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
            {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
            {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
            {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
            {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
        ]
        
        for g in range(2):  # Two blocks for 2x6
            for s in range(7):
                r = 7 * g + s
                step = strassen_steps[s]
                for local_idx in range(4):
                    u_val = step['u'][local_idx]
                    if u_val != 0:
                        U[local_idx, r] = u_val
                    v_val = step['v'][local_idx]
                    if v_val != 0:
                        V[local_idx * 6 + 2*g, r] = v_val
                    w_val = step['w'][local_idx]
                    if w_val != 0:
                        W[(2*g + local_idx//2) * 2 + local_idx%2, r] = w_val
        
        # Last 6 ranks: random
        U[:, 14:] = np.random.randn(nm, rank_try - 14) * 0.5
        V[:, 14:] = np.random.randn(mp, rank_try - 14) * 0.5
        W[:, 14:] = np.random.randn(pn, rank_try - 14) * 0.5
        
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Try multiple initialization strategies
    for attempt in range(30):
        if time.time() > deadline - 10:
            break
        
        # Alternate between random and structured initializations
        if attempt % 3 == 0:
            # Pure random with different scales
            np.random.seed(200 + attempt)
            scale = [0.5, 1.0, 2.0][attempt % 3]
            x0 = np.random.randn(nm * rank_try + mp * rank_try + pn * rank_try) * scale
        elif attempt % 3 == 1:
            # Structured initialization
            x0 = generate_structured_init(300 + attempt)
        else:
            # Random with sparsity
            np.random.seed(400 + attempt)
            x0 = np.random.randn(nm * rank_try + mp * rank_try + pn * rank_try)
            x0[np.abs(x0) < 0.5] = 0  # Add sparsity
        
        try:
            result = minimize(
                objective, 
                x0, 
                args=(rank_try,),
                method='L-BFGS-B', 
                jac=gradient,
                options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_error:
                best_error = result.fun
                best_result = result.x
                
                # If we found a very good solution, do a high-precision polish pass
                if best_error < 1e-6:
                    result2 = minimize(
                        objective, 
                        best_result, 
                        args=(rank_try,),
                        method='L-BFGS-B', 
                        jac=gradient,
                        options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    if result2.fun < best_error:
                        best_error = result2.fun
                        best_result = result2.x
                        
        except Exception as e:
            pass
>>>>>>> REPLACE

