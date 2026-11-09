<<<<<<< SEARCH
def _basin_hopping_chain(rho, x0, rng, max_iter=100, temp_start=0.1, temp_end=0.01):
    """
    Basin hopping with cooling temperature.
    Returns (best_x, best_e).
    """
    best_x = x0.copy()
    best_e = _morse_energy(rho, best_x)
    cur_x = best_x.copy()
    cur_e = best_e
    
    # Initial relaxation
    e_rel, x_rel = relax(x0, 'morse', rho)
    if e_rel < 1e8:
        cur_x = x_rel
        cur_e = e_rel
        if cur_e < best_e:
            best_e = cur_e
            best_x = x_rel.copy()

    for i in range(max_iter):
        if _remaining() < 2.0:
            break
        
        # Cooling schedule
        t = i / max_iter
        temp = temp_start * (temp_end / temp_start) ** t
        
        # Perturb
        # Scale increases slightly as we cool to allow larger jumps initially? 
        # Actually, usually scale is constant or increases. Let's use a moderate scale.
        scale = 0.2 + 0.1 * np.sin(np.pi * t) # Vary scale a bit
        x_new = _perturb(cur_x, rng, scale)
        
        # Relax
        e_rel, x_rel = relax(x_new, 'morse', rho)
        
        if e_rel > 1e8:
            # Try to fix and retry once
            x_fixed = _validate_and_fix(x_rel)
            e_rel, x_rel = relax(x_fixed, 'morse', rho)
            if e_rel > 1e8:
                continue

        dE = e_rel - cur_e
        if dE < 0 or rng.random() < np.exp(-dE / temp):
            cur_x = x_rel
            cur_e = e_rel
            if cur_e < best_e:
                best_e = cur_e
                best_x = x_rel.copy()
                
    return best_x, best_e
=======
def _basin_hopping_chain(rho, x0, rng, max_iter=200, temp_start=0.15, temp_end=0.005):
    """
    Basin hopping with adaptive cooling and occasional restarts.
    Returns (best_x, best_e).
    """
    best_x = x0.copy()
    best_e = _morse_energy(rho, best_x)
    cur_x = best_x.copy()
    cur_e = best_e
    
    # Initial relaxation
    e_rel, x_rel = relax(x0, 'morse', rho)
    if e_rel < 1e8:
        cur_x = x_rel
        cur_e = e_rel
        if cur_e < best_e:
            best_e = cur_e
            best_x = x_rel.copy()

    no_improve_count = 0
    max_no_improve = 30
    
    for i in range(max_iter):
        if _remaining() < 2.0:
            break
        
        # Adaptive cooling schedule with occasional reheating
        t = i / max_iter
        # Use a steeper cooling curve
        temp = temp_start * (temp_end / temp_start) ** (t ** 0.8)
        
        # Reheat if stuck
        if no_improve_count > max_no_improve:
            temp = temp_start * 0.5
            no_improve_count = 0
        
        # Perturb with adaptive scale
        # Scale decreases slightly as we cool, but with some randomness
        base_scale = 0.25 - 0.1 * t
        scale = base_scale * (0.8 + 0.4 * rng.random())
        x_new = _perturb(cur_x, rng, scale)
        
        # Relax
        e_rel, x_rel = relax(x_new, 'morse', rho)
        
        if e_rel > 1e8:
            # Try to fix and retry once
            x_fixed = _validate_and_fix(x_rel)
            e_rel, x_rel = relax(x_fixed, 'morse', rho)
            if e_rel > 1e8:
                continue

        dE = e_rel - cur_e
        # Metropolis criterion with adaptive acceptance
        if dE < 0:
            cur_x = x_rel
            cur_e = e_rel
            no_improve_count = 0
            if cur_e < best_e:
                best_e = cur_e
                best_x = x_rel.copy()
        else:
            # Accept with probability exp(-dE/temp), but be more accepting early on
            accept_prob = np.exp(-dE / temp)
            if i < max_iter * 0.3:
                accept_prob = max(accept_prob, 0.3)  # Higher acceptance early
            if rng.random() < accept_prob:
                cur_x = x_rel
                cur_e = e_rel
                no_improve_count = 0
            else:
                no_improve_count += 1
                
    return best_x, best_e
>>>>>>> REPLACE

