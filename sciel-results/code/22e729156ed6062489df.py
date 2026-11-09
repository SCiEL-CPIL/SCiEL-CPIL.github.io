import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 295.0  # Safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() - start_time < time_limit

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            if not time_ok():
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                v = float(func(x))
            except Exception:
                return best_f
            if not np.isfinite(v):
                v = best_f
                return v
            calls += 1
            if v < best_f:
                best_f = v
                best_x = x.copy()
            return v

        # 1. Initialization with Latin Hypercube + Origin
        init_points = []
        init_points.append(np.zeros(dim))
        
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=dim, seed=42)
            points = sampler.random(n=40)
            # Scale from [0,1] to [-4,4]
            points = points * 8.0 - 4.0
            init_points.extend(points)
        except ImportError:
            init_points.extend(rng.uniform(-4, 4, (40, dim)))

        # Evaluate init
        init_values = []
        for p in init_points:
            if not time_ok() or calls >= budget - 100:
                break
            v = wrapped_func(p)
            init_values.append(v)
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)

        sorted_idx = np.argsort(init_values)
        # Get top 5 starting points for multi-start
        top_starts = []
        for idx in sorted_idx[:5]:
            top_starts.append(init_points[idx].copy())

        # 2. Multi-Start Nelder-Mead
        # We use the top 5 starting points for multi-start
        remaining_budget = budget - calls - 10
        if remaining_budget > 0:
            n_restarts = min(5, len(top_starts))
            if n_restarts == 0:
                top_starts = [best_x.copy()]
                n_restarts = 1
            
            # Distribute budget among restarts
            # Reserve some for final polish
            polish_budget = 50
            search_budget = remaining_budget - polish_budget
            if search_budget < 10:
                search_budget = remaining_budget
                polish_budget = 0
                
            budget_per_restart = max(10, search_budget // n_restarts)
            
            for i in range(n_restarts):
                if not time_ok() or calls >= budget - 10:
                    break
                
                x0 = top_starts[i].copy()
                
                # Run Nelder-Mead
                try:
                    from scipy.optimize import minimize
                    # Use a custom callback to stop if budget is hit
                    # Nelder-Mead maxfev is the number of function calls
                    res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                                   options={'maxiter': 200, 'maxfev': budget_per_restart,
                                            'xatol': 1e-6, 'fatol': 1e-8,
                                            'adaptive': True})
                    # wrapped_func updates best_x and best_f internally
                except Exception:
                    pass
        
        # 3. Final Polish
        if best_x is not None and calls < budget - 5 and time_ok():
            # Run a tight Nelder-Mead on the global best
            try:
                from scipy.optimize import minimize
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': 100, 'maxfev': 50,
                                        'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True})
            except Exception:
                pass

        return best_x, best_f
