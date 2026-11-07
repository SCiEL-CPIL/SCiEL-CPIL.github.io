import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_f = float('inf')
        best_x = None
        start_time = time.time()
        deadline = start_time + 285.0

        # Wrapper to count calls and enforce deadline
        def wrapper(x):
            nonlocal calls_made
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls_made >= total_budget:
                raise TimeoutError("Budget exceeded")
            calls_made += 1
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(12345)

        def halton_sequence(i, b):
            f = 1.0
            r = 0.0
            while i > 0:
                f /= b
                r += f * (i % b)
                i //= b
            return r

        def halton_point(index, dim):
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            point = np.zeros(dim)
            for d in range(dim):
                point[d] = halton_sequence(index + d + 1, primes[d])
            return point

        # Phase 1: Initial Global Search
        # Use Halton for coverage and a few random points for diversity
        n_global = 40
        
        for i in range(n_global):
            if calls_made >= total_budget or time.time() > deadline:
                break
            if i < 30:
                x_unit = halton_point(i + 1, dim)
                x = x_unit * 10.0 - 5.0
            else:
                x = rng.uniform(-5.0, 5.0, dim)
            
            try:
                fx = wrapper(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            except TimeoutError:
                break

        if best_x is None:
            return np.zeros(dim), best_f

        # Phase 2: Hierarchical Multi-Scale Stochastic Tunneling
        # Start with a medium step size
        sigma = 1.0
        min_sigma = 1e-6
        max_sigma = 2.0
        
        stagnation_count = 0
        consecutive_rejections = 0
        tunnel_threshold = 40 # If stuck for this many rejections, try a "tunnel" (scale reset)
        tunnel_sigma = 0.5   # Scale to use for tunneling
        
        while calls_made < total_budget and time.time() < deadline:
            # Generate trial point
            x_trial = best_x + sigma * rng.standard_normal(dim)
            
            try:
                fx_trial = wrapper(x_trial)
            except TimeoutError:
                break
            
            if fx_trial < best_f:
                # Improvement
                best_f = fx_trial
                best_x = x_trial.copy()
                stagnation_count = 0
                consecutive_rejections = 0
                # Expand sigma slightly to explore around new best
                sigma = min(sigma * 1.1, max_sigma)
            else:
                # No improvement
                consecutive_rejections += 1
                
                # Adaptive shrinking
                if consecutive_rejections > 5:
                    sigma = max(sigma * 0.8, min_sigma)
                    consecutive_rejections = 0 # Reset rejection counter for shrinking logic
                    
                # Stagnation detection for tunneling
                if fx_trial > best_f:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                
                # Tunneling mechanism:
                # If we are stuck (no improvement for a long time) and sigma is still relatively large or medium,
                # we might be in a local minimum. A "tunnel" involves jumping with a specific scale 
                # to potentially escape the local basin.
                if stagnation_count > tunnel_threshold and sigma > 0.2:
                    # Reset sigma to a medium value to attempt a larger jump from the current best
                    # This effectively restarts the local search at a coarser scale
                    sigma = tunnel_sigma
                    stagnation_count = 0

        return best_x, best_f
