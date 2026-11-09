import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        timeout = 280  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        rng = np.random.default_rng(42)
        
        # 1. Initialization: Diverse sampling including corners and random
        # Use ~10% of budget for initial population
        init_pop_size = max(20, self.budget // 10)
        
        X_init = []
        # 50% random
        for _ in range(init_pop_size // 2):
            X_init.append(rng.random(self.dim) * span + lo)
        # 50% boundary-focused (helps find Pareto extremes)
        for _ in range(init_pop_size - len(X_init)):
            x = rng.random(self.dim)
            # Force some dimensions to be near bounds (0 or 1 in normalized space)
            mask = rng.random(self.dim) < 0.3
            if np.any(mask):
                x[mask] = rng.random(np.sum(mask))
            x = x * span + lo
            X_init.append(x)
            
        X_init = np.array(X_init)
        
        # Evaluate initial population
        F_init = np.array([problem(x) for x in X_init])
        used_evals = len(X_init)
        
        # Build initial Pareto Front
        # We maintain a list of (x, f) pairs on the front
        pareto_front = []
        
        def is_dominated(f_new, front_f_list):
            for f_old in front_f_list:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return True
            return False

        def dominates(f_a, f_b):
            return np.all(f_a <= f_b) and np.any(f_a < f_b)

        for x, f in zip(X_init, F_init):
            # Check if f is dominated by existing front
            dominated = False
            for _, f_old in pareto_front:
                if dominates(f_old, f):
                    dominated = True
                    break
            if not dominated:
                # Add new point
                pareto_front.append((x, f))
                # Remove points from front that are dominated by f
                pareto_front = [
                    (x_p, f_p) for x_p, f_p in pareto_front 
                    if not dominates(f, f_p)
                ]
        
        # 2. Main Loop: Adaptive Mutation
        # Start with a relatively large sigma to explore, decay slowly to exploit
        sigma = 0.15 
        min_sigma = 0.005
        decay_rate = 0.995
        
        # Track stagnation
        stagnation_count = 0
        max_stagnation = 100
        
        while used_evals < self.budget:
            if time.time() - start_time > timeout:
                break
                
            if not pareto_front:
                # Fallback: random restart
                x = rng.random(self.dim) * span + lo
                f = problem(x)
                pareto_front.append((x, f))
                used_evals += 1
                continue

            # Selection: Pick a parent from the Pareto front
            # Uniform random selection is simple and effective for small fronts
            parent_idx = rng.integers(len(pareto_front))
            parent_x, parent_f = pareto_front[parent_idx]
            
            # Mutation: Gaussian noise scaled by sigma and span
            noise = rng.standard_normal(self.dim) * sigma * span
            child_x = parent_x + noise
            child_x = np.clip(child_x, lo, hi)
            
            # Evaluation
            child_f = problem(child_x)
            used_evals += 1
            
            # Update Pareto Front
            dominated = False
            for _, f_old in pareto_front:
                if dominates(f_old, child_f):
                    dominated = True
                    break
            
            if not dominated:
                # Add child
                pareto_front.append((child_x, child_f))
                # Remove dominated points
                pareto_front = [
                    (x_p, f_p) for x_p, f_p in pareto_front 
                    if not dominates(child_f, f_p)
                ]
                
                # If we found a new non-dominated point, we are making progress
                stagnation_count = 0
                # Slight decay to refine search
                sigma *= decay_rate
                if sigma < min_sigma:
                    sigma = min_sigma
            else:
                # No improvement
                stagnation_count += 1
                # If stuck for a while, increase sigma to escape local optima
                if stagnation_count > max_stagnation:
                    sigma = min(0.2, sigma * 1.5)
                    stagnation_count = 0

        # Return the Pareto front (value is ignored, but must not crash)
        return pareto_front
