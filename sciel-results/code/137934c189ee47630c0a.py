import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # Phase 1: Global Search (Random Sampling)
        # Use 15% of budget to find a good basin
        init_budget = max(10, int(budget * 0.15))
        
        # Ensure we don't exceed budget
        if init_budget >= budget:
            init_budget = budget - 1 # Leave at least 1 for NM init? Or just stop.
            # If budget is very small, just do random search
            
        init_pop = rng.uniform(-5, 5, (init_budget, n))
        init_fits = np.array([func(x) for x in init_pop])
        evals_used = init_budget
        
        best_idx = np.argmin(init_fits)
        best_x = init_pop[best_idx].copy()
        best_f = init_fits[best_idx]
        
        if evals_used >= budget:
            return best_x, best_f
            
        # Phase 2: Local Refinement (Nelder-Mead)
        # Initialize simplex around best_x
        alpha = 0.2
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_x
        simplex_fits = np.zeros(n + 1)
        simplex_fits[0] = best_f
        
        for i in range(1, n + 1):
            simplex[i] = best_x + alpha * rng.standard_normal(n)
            simplex[i] = np.clip(simplex[i], -5, 5)
            
            if evals_used < budget:
                simplex_fits[i] = func(simplex[i])
                evals_used += 1
            else:
                # If we run out of budget here, we can't complete simplex init
                # Just return best so far
                break
        
        if evals_used >= budget:
            return best_x, best_f
            
        # Nelder-Mead Loop
        alpha_nm = 1.0  # Reflection
        gamma_nm = 2.0  # Expansion
        rho_nm = 0.5    # Contraction
        sigma_nm = 0.5  # Shrink
        
        while evals_used < budget:
            # Sort simplex by fitness
            order = np.argsort(simplex_fits)
            simplex = simplex[order]
            simplex_fits = simplex_fits[order]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:n], axis=0)
            worst = simplex[-1]
            worst_f = simplex_fits[-1]
            best_f_simplex = simplex_fits[0]
            second_best_f = simplex_fits[1]
            
            # 1. Reflection
            x_r = centroid + alpha_nm * (centroid - worst)
            x_r = np.clip(x_r, -5, 5)
            
            if evals_used >= budget: break
            f_r = func(x_r)
            evals_used += 1
            
            if f_r < best_f_simplex:
                # 2. Expansion
                if evals_used >= budget: break
                x_e = centroid + gamma_nm * (centroid - worst)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                evals_used += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_fits[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_fits[-1] = f_r
            elif f_r < second_best_f:
                simplex[-1] = x_r
                simplex_fits[-1] = f_r
            else:
                # 3. Contraction
                if f_r < worst_f:
                    # Outside contraction
                    x_c = centroid + rho_nm * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                else:
                    # Inside contraction
                    x_c = centroid - rho_nm * (centroid - worst)
                    x_c = np.clip(x_c, -5, 5)
                    
                if evals_used >= budget: break
                f_c = func(x_c)
                evals_used += 1
                
                if f_c < min(f_r, worst_f):
                    simplex[-1] = x_c
                    simplex_fits[-1] = f_c
                else:
                    # 4. Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        
                        if evals_used >= budget: break
                        simplex_fits[i] = func(simplex[i])
                        evals_used += 1
                    
                    if evals_used >= budget:
                        break
            
            # Update global best
            current_min_idx = np.argmin(simplex_fits)
            if simplex_fits[current_min_idx] < best_f:
                best_f = simplex_fits[current_min_idx]
                best_x = simplex[current_min_idx].copy()
                
        return best_x, best_f
