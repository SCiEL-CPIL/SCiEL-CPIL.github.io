<<<<<<< SEARCH
def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0
    
    def time_ok():
        return time.time() < DEADLINE
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.67, 1.4, 0.71]
    
    for W_ratio in aspect_ratios:
        if not time_ok():
            break
        
        W = W_ratio
        H = 2.0 - W
        if W <= 0 or H <= 0:
            continue
        
        # Try multiple initializations
        inits = []
        
        # Hexagonal
        for n_rows in [3, 4, 5]:
            inits.append(_hex_init(W, H, n_rows))
        
        # Square grid
        inits.append(_square_grid_init(W, H))
        
        # Random
        for _ in range(3):
            inits.append(_random_init(W, H))
        
        for x0 in inits:
            if not time_ok():
                break
            
            # Perturb slightly
            x0_perturbed = x0 + np.random.normal(0, 0.01, x0.shape)
            
            res_x, res_sum = _optimize_from(x0_perturbed, W, H)
            
            if res_sum > best_sum:
                best_sum = res_sum
                best_arr = res_x.copy()
                best_W = W
                best_H = H
    
    if best_arr is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Final validation and clamping
    W, H = best_W, best_H
    cx = np.clip(best_arr[:, 0], 0.0, W)
    cy = np.clip(best_arr[:, 1], 0.0, H)
    r = np.maximum(best_arr[:, 2], 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    
    result = np.column_stack([cx, cy, r])
    return result
=======
def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0
    
    def time_ok():
        return time.time() < DEADLINE
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.67, 1.4, 0.71]
    
    for W_ratio in aspect_ratios:
        if not time_ok():
            break
        
        W = W_ratio
        H = 2.0 - W
        if W <= 0 or H <= 0:
            continue
        
        # Try multiple initializations
        inits = []
        
        # Hexagonal
        for n_rows in [3, 4, 5]:
            inits.append(_hex_init(W, H, n_rows))
        
        # Square grid
        inits.append(_square_grid_init(W, H))
        
        # Random
        for _ in range(3):
            inits.append(_random_init(W, H))
        
        for x0 in inits:
            if not time_ok():
                break
            
            # Perturb slightly
            x0_perturbed = x0 + np.random.normal(0, 0.01, x0.shape)
            
            res_x, res_sum = _optimize_from(x0_perturbed, W, H)
            
            if res_sum > best_sum:
                best_sum = res_sum
                best_arr = res_x.copy()
                best_W = W
                best_H = H
    
    if best_arr is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Final validation and clamping
    W, H = best_W, best_H
    cx = np.clip(best_arr[:, 0], 0.0, W)
    cy = np.clip(best_arr[:, 1], 0.0, H)
    r = np.maximum(best_arr[:, 2], 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    
    result = np.column_stack([cx, cy, r])
    return result
>>>>>>> REPLACE
