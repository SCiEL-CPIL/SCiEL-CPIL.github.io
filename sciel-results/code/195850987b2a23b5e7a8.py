<<<<<<< SEARCH
    def simulated_annealing(p, current_cost, end_time, seed=42):
        """
        Perform Simulated Annealing to escape local minima.
        """
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        
        rng = np.random.RandomState(seed)
        
        # Estimate initial temperature
        deltas = []
        for _ in range(20):
            i, j = rng.choice(n, 2, replace=False)
            d = compute_swap_delta(i, j, curr_p)
            deltas.append(abs(d))
        avg_delta = np.mean(deltas) if deltas else 1.0
        if avg_delta < 1e-6:
            avg_delta = 1.0
            
        T = avg_delta * 15.0
        cooling_rate = 0.9995
        min_T = 1e-5
        
        while time.time() < end_time:
            if T < min_T:
                break
                
            # Generate random move
            i, j = rng.choice(n, 2, replace=False)
            
            delta = compute_swap_delta(i, j, curr_p)
            
            # Acceptance criterion
            if delta < 0:
                accept = True
            else:
                # Avoid overflow in exp
                if T > 0:
                    prob = np.exp(-delta / T)
                else:
                    prob = 0.0
                accept = rng.random_sample() < prob
            
            if accept:
                curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                curr_cost += delta
                
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_p = curr_p.copy()
            
            T *= cooling_rate
        
        return best_p, best_cost
=======
    def simulated_annealing(p, current_cost, end_time, seed=42):
        """
        Perform Simulated Annealing to escape local minima.
        Uses adaptive cooling: slower cooling when improving, faster when stuck.
        """
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        
        rng = np.random.RandomState(seed)
        
        # Estimate initial temperature
        deltas = []
        for _ in range(20):
            i, j = rng.choice(n, 2, replace=False)
            d = compute_swap_delta(i, j, curr_p)
            deltas.append(abs(d))
        avg_delta = np.mean(deltas) if deltas else 1.0
        if avg_delta < 1e-6:
            avg_delta = 1.0
            
        T = avg_delta * 20.0
        base_cooling = 0.9992
        min_T = 1e-6
        
        # Track recent improvements for adaptive cooling
        recent_improvements = 0
        no_improve_count = 0
        last_best_cost = best_cost
        
        while time.time() < end_time:
            if T < min_T:
                break
                
            # Generate random move
            i, j = rng.choice(n, 2, replace=False)
            
            delta = compute_swap_delta(i, j, curr_p)
            
            # Acceptance criterion
            if delta < 0:
                accept = True
            else:
                # Avoid overflow in exp
                if T > 0:
                    ratio = delta / T
                    if ratio > 700:  # exp(-700) underflows
                        prob = 0.0
                    else:
                        prob = np.exp(-ratio)
                else:
                    prob = 0.0
                accept = rng.random_sample() < prob
            
            if accept:
                curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                curr_cost += delta
                
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_p = curr_p.copy()
                    recent_improvements += 1
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            # Adaptive cooling: slow down if improving, speed up if stuck
            if recent_improvements > 0:
                cooling = base_cooling * 0.999  # slower cooling
            elif no_improve_count > 5000:
                cooling = base_cooling * 1.005  # faster cooling when stuck
            else:
                cooling = base_cooling
            
            T *= cooling
        
        return best_p, best_cost
>>>>>>> REPLACE

