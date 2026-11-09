import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(n)
        best_f = np.inf
        evals_used = 0
        
        # Multi-start strategy
        # Run NM for 200 evals, then restart if no significant improvement
        # Or just run 2 independent NM runs of 250 evals each.
        
        num_restarts = 2
        budget_per_run = total_budget // num_restarts
        
        for r in range(num_restarts):
            # Initialize start point
            if r == 0:
                x_start = np.zeros(n)
            else:
                # Random start in [-4, 4]
                x_start = rng.uniform(-4, 4, n)
            
            # Run NM
            # We need to track evals used in this run
            run_budget = budget_per_run
            
            # Create simplex
            x0 = x_start
            f0 = func(x0)
            evals_used += 1
            run_budget -= 1
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
                
            delta = 0.5
            simplex = [x0]
            fvals = [f0]
            
            for i in range(n):
                x_i = x0.copy()
                x_i[i] += delta
                simplex.append(x_i)
                fvals.append(func(x_i))
                evals_used += 1
                run_budget -= 1
                if fvals[-1] < best_f:
                    best_f = fvals[-1]
                    best_x = simplex[-1].copy()
                    
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            while run_budget > 0:
                # Sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                evals_used += 1
                run_budget -= 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                    
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    evals_used += 1
                    run_budget -= 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                        
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (centroid - simplex[-1])
                        
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    evals_used += 1
                    run_budget -= 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                        
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            evals_used += 1
                            run_budget -= 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()
                            if run_budget <= 0:
                                break
                        if run_budget <= 0:
                            break
                            
        return best_x, best_f
