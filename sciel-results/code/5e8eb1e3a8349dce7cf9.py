import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290.0  # Safety margin before 300s kill
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

        # 1. Initialization: Latin Hypercube + structured points for Katsuura
        init_points = []
        init_points.append(np.zeros(dim))
        
        # Add points near boundaries and structured grid for Katsuura's multimodality
        for axis in range(dim):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                p = np.zeros(dim)
                p[axis] = val
                init_points.append(p)
        
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=dim, seed=42)
            n_lh = max(10, budget // 50)  # ~10-20 points
            points = sampler.random(n=n_lh)
            points = points * 8.0 - 4.0  # Scale to [-4, 4]
            init_points.extend(points)
        except ImportError:
            init_points.extend(rng.uniform(-4, 4, (15, dim)))

        # Deduplicate and limit init points
        init_points = np.array(init_points)
        init_points = np.clip(init_points, -5.0, 5.0)
        
        # Keep unique points (within tolerance)
        unique_points = [init_points[0]]
        for p in init_points[1:]:
            if all(np.linalg.norm(p - u) > 0.1 for u in unique_points):
                unique_points.append(p)
        init_points = np.array(unique_points)
        
        # Limit to reasonable number
        max_init = min(len(init_points), 40)
        init_points = init_points[:max_init]

        # Evaluate init
        init_values = []
        for p in init_points:
            if not time_ok() or calls >= budget - 50:
                break
            v = wrapped_func(p)
            init_values.append(v)
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)

        sorted_idx = np.argsort(init_values)
        # Get top starting points for multi-start CMA-ES
        top_starts = []
        for idx in sorted_idx[:min(5, len(sorted_idx))]:
            top_starts.append(init_points[idx].copy())

        # 2. Multi-Start CMA-ES
        use_cma = False
        try:
            import cma
            use_cma = True
        except ImportError:
            use_cma = False

        if use_cma:
            n_starts = min(4, len(top_starts))
            # Allocate budget: give more to first start, less to subsequent
            base_budget = (budget - calls - 30) // n_starts
            
            for i in range(n_starts):
                if not time_ok() or calls >= budget - 20:
                    break
                
                x0 = top_starts[i].copy()
                
                # Adapt sigma based on initial best
                if i == 0:
                    sigma0 = 1.5
                else:
                    sigma0 = 0.8
                
                bounds = [(-5.0, 5.0)] * dim
                popsize = 4 if dim <= 5 else 6
                
                try:
                    es = cma.CMA(x0, sigma0, {
                        'bounds': bounds,
                        'maxiter': 1000,
                        'seed': 42 + i,
                        'verbose': -9,
                        'tolopt': 1e-14,
                        'tolfun': 1e-14,
                        'tolfunint': 1e-14,
                        'popsize': popsize,
                        'CMA_adapt_sigma': True,
                    })
                    
                    while not es.stop() and calls < budget - 15 and time_ok():
                        solutions = es.ask()
                        fitnesses = []
                        valid_count = 0
                        for sol in solutions:
                            if calls >= budget - 15 or not time_ok():
                                break
                            v = wrapped_func(sol)
                            fitnesses.append(v)
                            valid_count += 1
                        
                        if valid_count == 0:
                            break
                            
                        # Only tell if we have enough valid solutions
                        if valid_count >= 2:
                            es.tell(solutions[:valid_count], fitnesses)
                        else:
                            break
                        
                except Exception:
                    pass
        
        # 3. Fallback / Additional (1+1)-ES for remaining budget
        if calls < budget - 10 and time_ok():
            # Use best_x as start for a refined (1+1)-ES
            x = best_x.copy()
            f_x = best_f
            sigma = 0.3
            no_improve = 0
            max_steps = min(budget - calls - 5, 100)
            
            for _ in range(max_steps):
                if not time_ok() or calls >= budget - 5:
                    break
                if sigma < 1e-10:
                    break
                    
                mutation = rng.normal(0, 1, dim)
                y = x + sigma * mutation
                y = np.clip(y, -5, 5)
                
                f_y = wrapped_func(y)
                
                if f_y < f_x - 1e-12:
                    x = y
                    f_x = f_y
                    sigma *= 1.03
                    no_improve = 0
                else:
                    sigma *= 0.85
                    no_improve += 1
                    if no_improve > 30:
                        break

        # 4. Final Polish with Nelder-Mead
        if best_x is not None and calls < budget - 3 and time_ok():
            try:
                from scipy.optimize import minimize
                # Use a tight NM starting from best
                nm_budget = budget - calls - 2
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': min(nm_budget, 80), 
                                        'xatol': 1e-10, 
                                        'fatol': 1e-14,
                                        'adaptive': True})
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
