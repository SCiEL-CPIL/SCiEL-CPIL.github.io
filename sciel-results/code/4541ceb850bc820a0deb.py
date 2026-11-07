import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        x_best = None
        fx_best = float('inf')
        calls = 0

        def track(x, fx):
            nonlocal x_best, fx_best
            if fx < fx_best:
                fx_best = fx
                x_best = x.copy()

        def safe_call(x):
            nonlocal calls
            if calls >= budget or time.time() > deadline:
                return None
            fx = func(x)
            calls += 1
            track(x, fx)
            return fx

        # Phase 1: Diverse initial exploration to find a good basin
        # Use ~10% of budget for initialization
        n_init = max(10, int(budget * 0.1))
        
        # Include center and random points
        init_points = [np.zeros(dim)]
        for _ in range(n_init - 1):
            init_points.append(rng.uniform(-5, 5, dim))

        for x0 in init_points:
            fx0 = safe_call(x0)
            if fx0 is None:
                break

        # Phase 2: (1+1)-ES Refinement
        # Use the remaining budget for a single sustained lineage from the best point found
        if x_best is not None:
            x = x_best.copy()
            fx = fx_best
            sigma = 1.0  # Start with a reasonable step size
            
            # Adaptive (1+1)-ES
            while calls < budget and time.time() < deadline:
                # Generate offspring
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = safe_call(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x = y
                    fx = fy
                    # Increase sigma on success, but cap it
                    sigma = min(sigma * 1.05, 5.0)
                else:
                    # Decrease sigma on failure
                    sigma = max(sigma * 0.95, 1e-6)
                    
                    # Early stopping if sigma is too small and no progress
                    if sigma < 1e-5:
                        break

        return x_best, fx_best
