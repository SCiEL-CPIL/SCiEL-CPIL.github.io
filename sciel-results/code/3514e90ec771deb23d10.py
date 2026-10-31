<<<<<<< SEARCH
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break
            
        # Run SA
        # Adjust steps based on remaining time
        time_left = deadline - time.time()
        steps = int(min(30000, time_left * 600)) # Rough estimate
        
        pos_sa, rad_sa = _sa_packing(pos_init, rad_init, steps=steps, seed=idx*100 + 42, deadline=deadline - 8)
        
        # Polish
        if time.time() < deadline - 4:
            pos_pol, rad_pol = _polish(pos_sa, rad_sa, deadline)
            
            # Validate and shrink
            pos_val, rad_val = _shrink_to_valid(pos_pol, rad_pol)
            
            if _is_valid(pos_val, rad_val):
                score = np.sum(rad_val)
                if score > best_score:
                    best_score = score
                    best_pos = pos_val.copy()
                    best_radii = rad_val.copy()
                    
    # If no valid solution found, fallback to a simple grid
=======
    # 6. Additional random inits with different seeds
    for seed in [123, 456, 789]:
        rng2 = np.random.RandomState(seed)
        pos_r = rng2.uniform(0.05, 0.95, (N, 2))
        rad_r = rng2.uniform(0.04, 0.12, N)
        inits.append((pos_r, rad_r))
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        time_left = deadline - time.time()
        steps = int(min(25000, time_left * 500))
        
        pos_sa, rad_sa = _sa_packing(pos_init, rad_init, steps=steps, seed=idx*100 + 42, deadline=deadline - 15)
        
        # Local refinement
        if time.time() < deadline - 10:
            pos_lr, rad_lr = _local_refine(pos_sa, rad_sa, n_iter=300, deadline=deadline - 10)
            pos_sa, rad_sa = pos_lr, rad_lr
        
        # Polish
        if time.time() < deadline - 5:
            pos_pol, rad_pol = _polish(pos_sa, rad_sa, deadline)
            
            pos_val, rad_val = _shrink_to_valid(pos_pol, rad_pol)
            
            if _is_valid(pos_val, rad_val):
                score = np.sum(rad_val)
                if score > best_score:
                    best_score = score
                    best_pos = pos_val.copy()
                    best_radii = rad_val.copy()
                    
    if best_pos is None:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                if i*5+j < N-1:
                    pos[i*5+j] = [(i+0.5)/5.0, (j+0.5)/5.0]
                    radii[i*5+j] = 0.09
        pos[N-1] = [0.5, 0.02]
        radii[N-1] = 0.01
        best_pos, best_radii = _shrink_to_valid(pos, radii)
        
    result = np.column_stack([best_pos, best_radii])
    return result
>>>>>>> REPLACE
