"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid Multi-Start (1+1)-ES Optimizer.
    
    Strategy:
    1. Initial Diversification: Sample a small number of diverse points (random + origin) 
       to identify a promising basin.
    2. Exploitation: Run an aggressive (1+1)-Evolution Strategy from the best initial point.
       The step size adapts quickly (increase on success, decrease on failure) to navigate 
       the landscape efficiently within the limited budget.
    3. Safety: Strict deadline and budget enforcement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget
        calls_made = 0

        best_x = None
        best_f = np.inf

        # Phase 1: Initial Diversification
        # We sample a few points to ensure we don't start in a terrible local minimum.
        # 1 origin + 5 random points = 6 calls.
        n_init_samples = 5
        init_candidates = [np.zeros(n)]
        for _ in range(n_init_samples):
            init_candidates.append(rng.uniform(-5, 5, n))

        for x in init_candidates:
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made = 1

        # Phase 2: Aggressive (1+1)-ES from the best initial point
        x = best_x.copy()
        fx = best_f
        
        # Initial step size: 1.0 is a reasonable default for [-5, 5]^5
        sigma = 1.0
        
        # Parameters for aggressive adaptation
        sigma_inc = 1.5
        sigma_dec = 0.87
        sigma_min = 0.01
        sigma_max = 10.0

        remaining_budget = total_budget - calls_made

        for _ in range(remaining_budget):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break

            # Generate offspring
            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Success: move to y, increase step size
                x, fx = y, fy
                sigma *= sigma_inc
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Failure: stay at x, decrease step size
                sigma *= sigma_dec

            # Clamp sigma to valid range
            sigma = max(sigma_min, min(sigma, sigma_max))

        return best_x, best_f
# EVOLVE-BLOCK-END
