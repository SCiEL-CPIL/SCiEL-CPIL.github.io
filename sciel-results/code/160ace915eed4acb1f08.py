import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 300.0
        dim = self.dim
        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return best_f
            if time.time() - start_time > time_limit - 5.0:
                return best_f
            x = np.clip(x, -5, 5)
            v = func(x)
            calls += 1
            if v < best_f:
                best_f = v
                best_x = x.copy()
            return v

        # 1. Initialization with Latin Hypercube + Origin
        init_points = []
        # Origin
        init_points.append(np.zeros(dim))
        
        # LHS for better coverage
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=dim, seed=42)
            points = sampler.random(n=20)
            # Scale from [0,1] to [-4,4]
            points = points * 8.0 - 4.0
            init_points.extend(points)
        except ImportError:
            # Fallback to random
            init_points.extend(rng.uniform(-4, 4, (20, dim)))

        # Evaluate init
        init_values = []
        for p in init_points:
            v = wrapped_func(p)
            init_values.append(v)
        
        # Sort
        sorted_idx = np.argsort(init_values)
        
        # 2. Multi-Start (1+1)-ES
        step_budgets = [150, 100, 80, 60, 40]
        
        for i, seed_idx in enumerate(sorted_idx[:5]):
            if calls >= self.budget - 10 or time.time() - start_time > time_limit - 10.0:
                break
            
            x = init_points[seed_idx].copy()
            f_x = init_values[seed_idx]  # Use cached value, no extra call
            
            sigma = 0.5
            no_improve = 0
            max_steps = step_budgets[i]
            
            for _ in range(max_steps):
                if calls >= self.budget - 5 or time.time() - start_time > time_limit - 5.0:
                    break
                if sigma < 1e-8:
                    break
                    
                # Adaptive sigma: increase on success, decrease on failure
                mutation = rng.normal(0, 1, dim)
                y = x + sigma * mutation
                y = np.clip(y, -5, 5)
                
                f_y = wrapped_func(y)
                
                if f_y < f_x - 1e-10:
                    x = y
                    f_x = f_y
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.85
                    no_improve += 1
                    if no_improve > 40:
                        break
            
        # 3. Final Polish with Nelder-Mead
        if best_x is not None and calls < self.budget - 10 and time.time() - start_time < time_limit - 15.0:
            try:
                from scipy.optimize import minimize
                # Use more iterations and tighter tolerances for better convergence
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
