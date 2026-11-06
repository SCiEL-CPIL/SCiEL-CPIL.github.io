<<<<<<< SEARCH
    # Multi-restart
    best_violations = float('inf')
    best_colors = None
    
    num_restarts = 20
    restart_budget = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 1:
            break
            
        # Greedy initial coloring
        colors = greedy_init()
        current_violations = count_violations()
        
        if current_violations == 0:
            return True, colors
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
        # Phase 1: Targeted fixing
        success, current_violations = targeted_fix(colors, current_violations, deadline)
        if success:
            return True, colors
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
        # Phase 2: Simulated Annealing
        T = 1.0
        min_T = 0.01
        phase_deadline = min(deadline - 1, time.time() + restart_budget * 0.7)
        stagnant_steps = 0
        
        while time.time() < phase_deadline:
            k = random.randint(0, m - 1)
            u = edge_u[k]
            v = edge_v[k]
            old_c = colors[k]
            new_c = random.randint(0, 4)
            
            if new_c == old_c:
                continue
                
            delta = compute_delta(k, old_c, new_c)
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                apply_change(k, old_c, new_c, colors)
                current_violations += delta
                
                if current_violations == 0:
                    return True, colors
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                    stagnant_steps = 0
                else:
                    stagnant_steps += 1
                    
                if stagnant_steps > 5000:
                    T = min(2.0, T * 1.5)
                    stagnant_steps = 0
            else:
                stagnant_steps += 1
                
            T *= 0.999
            if T < min_T:
                T = min_T
                
        # Phase 3: Final targeted fix
        success, current_violations = targeted_fix(colors, current_violations, deadline)
        if success:
            return True, colors
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
    if best_violations == 0 and best_colors is not None:
        return True, best_colors
    return False, best_colors if best_colors else []
=======
    # Multi-restart with adaptive strategy
    best_violations = float('inf')
    best_colors = None
    
    # Dynamic number of restarts based on time
    elapsed = time.time() - (deadline - (deadline - time.time()))
    remaining = deadline - time.time()
    
    # Start with fewer, longer restarts for larger n
    if n >= 24:
        num_restarts = 8
    elif n >= 22:
        num_restarts = 12
    else:
        num_restarts = 16
    
    restart_budget = remaining / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 1:
            break
            
        # Greedy initial coloring
        colors = greedy_init()
        current_violations = count_violations()
        
        if current_violations == 0:
            return True, colors
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
        # Phase 1: Targeted fixing (more aggressive)
        fix_deadline = min(deadline - 1, time.time() + restart_budget * 0.3)
        success, current_violations = targeted_fix(colors, current_violations, fix_deadline)
        if success:
            return True, colors
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
        # Phase 2: Simulated Annealing
        T = 0.5
        min_T = 0.05
        phase_deadline = min(deadline - 1, time.time() + restart_budget * 0.6)
        stagnant_steps = 0
        best_in_sa = current_violations
        
        while time.time() < phase_deadline:
            k = random.randint(0, m - 1)
            u = edge_u[k]
            v = edge_v[k]
            old_c = colors[k]
            new_c = random.randint(0, 4)
            
            if new_c == old_c:
                continue
                
            delta = compute_delta(k, old_c, new_c)
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                apply_change(k, old_c, new_c, colors)
                current_violations += delta
                
                if current_violations == 0:
                    return True, colors
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                    stagnant_steps = 0
                    best_in_sa = current_violations
                elif current_violations < best_in_sa:
                    stagnant_steps = 0
                    best_in_sa = current_violations
                else:
                    stagnant_steps += 1
                    
                if stagnant_steps > 3000:
                    T = min(3.0, T * 1.8)
                    stagnant_steps = 0
            else:
                stagnant_steps += 1
                
            T *= 0.998
            if T < min_T:
                T = min_T
                
        # Phase 3: Final targeted fix
        success, current_violations = targeted_fix(colors, current_violations, deadline)
        if success:
            return True, colors
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
    if best_violations == 0 and best_colors is not None:
        return True, best_colors
    return False, best_colors if best_colors else []
>>>>>>> REPLACE
