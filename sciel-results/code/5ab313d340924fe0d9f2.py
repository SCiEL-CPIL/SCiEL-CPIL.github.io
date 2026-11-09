    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Initialize at centroid as per directive
        x0 = np.zeros(dim)
        f0 = func(x0)
        
        best_x = x0.copy()
        best_f = f0
        
        # Simplex initialization
        # Size of the simplex
        simplex_size = 0.5
        # We need dim+1 points. 
        # Point 0 is x0.
        # Points 1..dim are x0 + simplex_size * e_i
        simplex = [x0.copy()]
        f_vals = [f0]
        
        for i in range(dim):
            x_i = x0.copy()
            x_i[i] += simplex_size
            x_i = np.clip(x_i, -5, 5)
            f_i = func(x_i)
            simplex.append(x_i)
            f_vals.append(f_i)
            if f_i < best_f:
                best_f = f_i
                best_x = x_i.copy()
                
        evals_used = 1 + dim # 1 + 5 = 6
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        while evals_used < budget and time.time() < deadline:
            # Check if we have enough budget for a full iteration (worst case: reflection + expansion/contraction + shrink)
            # Max calls in one iteration: 1 (refl) + 1 (exp/contr) + dim (shrink) = dim + 2
            # To be safe, if we are close to budget, break.
            if evals_used + dim + 2 > budget:
                break
                
            # Sort simplex by function value
            order = np.argsort(f_vals)
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]
            
            # Check for convergence or stagnation
            # If the range of f_vals is very small, we might be stuck
            if max(f_vals) - min(f_vals) < 1e-8:
                break
                
            x_worst = simplex[-1]
            f_worst = f_vals[-1]
            x_best = simplex[0]
            f_best = f_vals[0]
            
            # Centroid of all but worst
            x_centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_reflected = x_centroid + alpha * (x_centroid - x_worst)
            x_reflected = np.clip(x_reflected, -5, 5)
            f_reflected = func(x_reflected)
            evals_used += 1
            
            if f_reflected < f_best:
                # Expansion
                x_expanded = x_centroid + gamma * (x_reflected - x_centroid)
                x_expanded = np.clip(x_expanded, -5, 5)
                f_expanded = func(x_expanded)
                evals_used += 1
                
                if f_expanded < f_reflected:
                    simplex[-1] = x_expanded
                    f_vals[-1] = f_expanded
                    if f_expanded < best_f:
                        best_f = f_expanded
                        best_x = x_expanded.copy()
                else:
                    simplex[-1] = x_reflected
                    f_vals[-1] = f_reflected
                    if f_reflected < best_f:
                        best_f = f_reflected
                        best_x = x_reflected.copy()
            elif f_reflected < f_vals[-2]:
                # Accept reflection
                simplex[-1] = x_reflected
                f_vals[-1] = f_reflected
                if f_reflected < best_f:
                    best_f = f_reflected
                    best_x = x_reflected.copy()
            else:
                # Contraction
                x_contracted = x_centroid + rho * (x_worst - x_centroid)
                x_contracted = np.clip(x_contracted, -5, 5)
                f_contracted = func(x_contracted)
                evals_used += 1
                
                if f_contracted < f_worst:
                    simplex[-1] = x_contracted
                    f_vals[-1] = f_contracted
                    if f_contracted < best_f:
                        best_f = f_contracted
                        best_x = x_contracted.copy()
                else:
                    # Shrink
                    # Only shrink if we have enough budget
                    # Shrink requires dim calls
                    if evals_used + dim > budget:
                        break
                    for i in range(1, len(simplex)):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_vals[i] = func(simplex[i])
                        evals_used += 1
                        if f_vals[i] < best_f:
                            best_f = f_vals[i]
                            best_x = simplex[i].copy()
                        
            # Update global best from current simplex (redundant if done above, but safe)
            for i, f in enumerate(f_vals):
                if f < best_f:
                    best_f = f
                    best_x = simplex[i].copy()
                    
        return best_x, best_f
