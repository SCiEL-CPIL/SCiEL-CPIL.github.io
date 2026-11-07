import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = None

    def __call__(self, func):
        start_time = time.time()
        self.deadline = start_time + 280  # 280s safety margin out of 330s
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def eval_func(x):
            nonlocal calls
            if calls >= budget:
                return best_f
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            f = float(func(x_clipped))
            return f
        
        # Nelder-Mead implementation
        def nelder_mead(x0, f0, max_iter):
            nonlocal best_x, best_f, calls
            
            n = dim
            # Initial simplex
            simplex = [x0.copy()]
            fvals = [f0]
            
            for i in range(n):
                x_i = x0.copy()
                if x0[i] == 0:
                    x_i[i] = 0.5
                else:
                    x_i[i] = x0[i] + 0.5 * np.sign(x0[i])
                x_i = np.clip(x_i, -5, 5)
                simplex.append(x_i)
                fvals.append(eval_func(x_i))
            
            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            
            for _ in range(max_iter):
                if time.time() > self.deadline:
                    break
                if calls >= budget:
                    break
                
                # Update best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = eval_func(x_r)
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = eval_func(x_e)
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
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = eval_func(x_c)
                    
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = eval_func(simplex[i])
                
                # Re-sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
            
            return simplex[0], fvals[0]
        
        # Multi-start strategy
        n_starts = 5
        budget_per_start = max(1, (budget - n_starts) // n_starts)
        
        for s in range(n_starts):
            if time.time() > self.deadline:
                break
            if calls >= budget:
                break
            
            if s == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4, 4, dim)
            
            x0 = np.clip(x0, -5, 5)
            f0 = eval_func(x0)
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            # Run Nelder-Mead
            x_opt, f_opt = nelder_mead(x0, f0, budget_per_start)
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
        
        # Final polish with smaller steps if time allows
        if best_x is not None and time.time() < self.deadline and calls < budget:
            x = best_x.copy()
            f = best_f
            sigma = 0.1
            for _ in range(min(50, budget - calls)):
                if time.time() > self.deadline:
                    break
                if calls >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)
                if fy < f:
                    x, f = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.7
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        return best_x, best_f
