import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        import time
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin
        
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # 1. Initialization: Find a good starting point
        # Use 10 random points
        init_count = 10
        if budget <= init_count:
            # Just do random search
            best_x = None
            best_f = np.inf
            for _ in range(budget):
                x = rng.uniform(-5, 5, n)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x
            return best_x, best_f
            
        X_init = rng.uniform(-5, 5, (init_count, n))
        F_init = np.array([func(x) for x in X_init])
        best_idx = np.argmin(F_init)
        x0 = X_init[best_idx].copy()
        f0 = F_init[best_idx]
        
        best_x = x0
        best_f = f0
        calls_used = init_count
        
        # 2. CMA-ES
        lambda_ = 10
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, sqrt(mu_eff/n - 1)) + c_sigma
        c_c = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
        c_1 = 2 / ((n+1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((n+2)**2 + mu_eff))
        
        sigma = 1.0  # Start with smaller sigma since we are in a basin
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        C_inv_sqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        x_mean = x0.copy()
        
        def update_covariance(C, C_inv, C_sqrt, C_inv_sqrt, weights, y_w, mu_eff, c_1, c_mu, n):
            # Standard CMA-ES update
            # This part is tricky to implement correctly from scratch without bugs.
            # Let's use a simpler adaptation: just adapt sigma and maybe diagonal covariance?
            # No, full CMA-ES is better.
            pass

        # Given the complexity and risk of bugs in manual CMA-ES, let's switch to a simpler but robust method:
        # **Multi-Start Nelder-Mead**
        # It's simpler, less prone to implementation errors, and works well on smooth multimodal functions once in a basin.
        
        # Let's restart the logic with Nelder-Mead.
        
        # We have best_x, best_f from init.
        # Run Nelder-Mead from best_x.
        
        # Nelder-Mead implementation
        # Simplex: n+1 points
        # Start with best_x and small perturbations
        
        def nelder_mead(func, x0, f0, n, budget_remaining, rng, time_limit_check):
            # Initialize simplex
            simplex = [x0.copy()]
            f_values = [f0]
            
            # Add n points with small step
            step = 0.5
            for i in range(n):
                x_i = x0.copy()
                x_i[i] += step
                simplex.append(x_i)
                f_i = func(x_i)
                f_values.append(f_i)
                
            # Track best
            best_x_local = x0
            best_f_local = f0
            
            # Main loop
            while budget_remaining > 0 and time_limit_check():
                # Sort simplex by function value
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]
                
                # Update global best
                if f_values[0] < best_f_local:
                    best_f_local = f_values[0]
                    best_x_local = simplex[0].copy()
                
                # Worst point
                x_worst = simplex[-1]
                f_worst = f_values[-1]
                
                # Centroid of all but worst
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_reflect = x_centroid + alpha * (x_centroid - x_worst)
                x_reflect = np.clip(x_reflect, -5, 5)
                f_reflect = func(x_reflect)
                budget_remaining -= 1
                
                if f_reflect < f_values[0]:
                    # Expansion
                    gamma = 2.0
                    x_expand = x_centroid + gamma * (x_centroid - x_worst)
                    x_expand = np.clip(x_expand, -5, 5)
                    f_expand = func(x_expand)
                    budget_remaining -= 1
                    
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        f_values[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        f_values[-1] = f_reflect
                elif f_reflect < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = x_reflect
                    f_values[-1] = f_reflect
                else:
                    # Contraction
                    rho = 0.5
                    if f_reflect < f_worst:
                        # Outside contraction
                        x_contract = x_centroid + rho * (x_reflect - x_centroid)
                    else:
                        # Inside contraction
                        x_contract = x_centroid - rho * (x_centroid - x_worst)
                        
                    x_contract = np.clip(x_contract, -5, 5)
                    f_contract = func(x_contract)
                    budget_remaining -= 1
                    
                    if f_contract < f_worst:
                        simplex[-1] = x_contract
                        f_values[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, n+1):
                            simplex[i] = x0 + 0.5 * (simplex[i] - x0)
                            f_values[i] = func(simplex[i])
                            budget_remaining -= 1
                            if budget_remaining <= 0:
                                break
                        
                        if budget_remaining <= 0:
                            break
                        
                        # Update best after shrink
                        order = np.argsort(f_values)
                        simplex = [simplex[i] for i in order]
                        f_values = [f_values[i] for i in order]
                        if f_values[0] < best_f_local:
                            best_f_local = f_values[0]
                            best_x_local = simplex[0].copy()
                        
                        # Check convergence
                        if np.std(f_values) < 1e-8:
                            break
                            
            return best_x_local, best_f_local, budget_remaining

        # Run Nelder-Mead
        x_best, f_best, remaining = nelder_mead(func, best_x, best_f, n, budget - calls_used, rng, lambda: time.time() > deadline)
        
        if f_best < best_f:
            best_f = f_best
            best_x = x_best
            
        return best_x, best_f
