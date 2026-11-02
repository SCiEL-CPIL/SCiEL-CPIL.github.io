<<<<<<< SEARCH
    def local_optimize(state, time_limit):
        """
        Run simulated annealing to improve state.
        """
        best_state = state.copy()
        best_score, _ = evaluate(best_state)
        
        T_start = 0.3
        T_end = 1e-6
        # Use a slower cooling schedule
        num_steps_estimate = 5000
        alpha = (T_end / T_start) ** (1.0 / num_steps_estimate)
        
        T = T_start
        current_state = best_state.copy()
        current_score = best_score
        
        while time.time() < time_limit - 1:
            # Generate neighbor
            new_state = current_state.copy()
            var_idx = random.randint(0, len(new_state)-1)
            
            if var_idx == 0:
                # Perturb aspect ratio
                new_state[0] += random.gauss(0, 0.03)
                if new_state[0] < 0.05 or new_state[0] > 0.95:
                    T *= alpha
                    if T < T_end:
                        T = T_end
                    continue
            else:
                circle_idx = (var_idx - 1) // 3
                comp = (var_idx - 1) % 3
                
                # Scale perturbation by current temperature
                scale = 0.05 * (T / T_start) + 0.005
                
                if comp == 0:  # x
                    new_state[var_idx] += random.gauss(0, scale)
                elif comp == 1:  # y
                    new_state[var_idx] += random.gauss(0, scale)
                else:  # r
                    new_state[var_idx] += random.gauss(0, scale * 0.5)
                    if new_state[var_idx] < 1e-4:
                        new_state[var_idx] = 1e-4
                    w, h = get_rect_dims(new_state[0])
                    max_r = 0.5 * min(w, h)
                    if new_state[var_idx] > max_r:
                        new_state[var_idx] = max_r
            
            new_score, new_penalty = evaluate(new_state)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta / T):
                current_state = new_state
                current_score = new_score
                if new_score > best_score:
                    best_state = new_state
                    best_score = new_score
            
            T *= alpha
            if T < T_end:
                T = T_end
        
        return best_state, best_score
=======
    def local_optimize(state, time_limit):
        """
        Run simulated annealing to improve state with adaptive step sizes.
        """
        best_state = state.copy()
        best_score, _ = evaluate(best_state)
        
        T_start = 0.5
        T_end = 1e-7
        # Adaptive cooling: estimate steps based on time
        steps_per_sec = 8000  # rough estimate
        est_steps = max(1000, int((time_limit - time.time()) * steps_per_sec))
        alpha = (T_end / T_start) ** (1.0 / est_steps)
        
        T = T_start
        current_state = best_state.copy()
        current_score = best_score
        
        # Track recent improvements for adaptive scaling
        recent_improvements = []
        no_improve_count = 0
        
        while time.time() < time_limit - 0.5:
            # Adaptive perturbation scale
            if len(recent_improvements) > 50:
                recent_improvements = recent_improvements[-50:]
            avg_improve = np.mean(recent_improvements) if recent_improvements else 0.01
            base_scale = 0.03 + 0.02 * (T / T_start)
            # If improving well, use larger steps; if stuck, try larger jumps
            if avg_improve > 0.001:
                scale_mult = 1.0
            elif no_improve_count > 200:
                scale_mult = 2.0  # escape local minima
            else:
                scale_mult = 0.8
            
            # Generate neighbor - choose which circle and component to perturb
            new_state = current_state.copy()
            
            # Weight toward perturbing circles with larger radii more
            w, h = get_rect_dims(new_state[0])
            radii = new_state[3:3+3*N:3]
            # Probability proportional to radius + small base
            probs = radii + 0.01
            probs /= probs.sum()
            
            if random.random() < 0.05:
                # Perturb aspect ratio
                new_state[0] += random.gauss(0, 0.02 * scale_mult)
                new_state[0] = np.clip(new_state[0], 0.05, 0.95)
            else:
                circle_idx = random.choice(N, p=probs)
                # Choose component: x, y, or r
                comp = random.randint(0, 2)
                var_idx = 1 + 3*circle_idx + comp
                
                scale = base_scale * scale_mult
                
                if comp == 0:  # x
                    new_state[var_idx] += random.gauss(0, scale)
                elif comp == 1:  # y
                    new_state[var_idx] += random.gauss(0, scale)
                else:  # r
                    new_state[var_idx] += random.gauss(0, scale * 0.4)
                    if new_state[var_idx] < 1e-4:
                        new_state[var_idx] = 1e-4
                    max_r = 0.5 * min(w, h)
                    if new_state[var_idx] > max_r:
                        new_state[var_idx] = max_r
            
            new_score, new_penalty = evaluate(new_state)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0 or (T > T_end * 10 and random.random() < np.exp(delta / T)):
                current_state = new_state
                current_score = new_score
                if new_score > best_score:
                    best_state = new_state
                    best_score = new_score
                    recent_improvements.append(delta)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            T *= alpha
            if T < T_end:
                T = T_end
        
        return best_state, best_score
>>>>>>> REPLACE

