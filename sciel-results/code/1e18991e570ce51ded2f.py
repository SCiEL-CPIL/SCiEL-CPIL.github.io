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
        best_x = np.zeros(dim)
        
        start_time = time.time()
        # Safety margin: 330s limit, use 325s
        deadline = start_time + 325.0

        def wrapper(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            calls_made += 1
            x = np.clip(x, -5, 5)
            try:
                fx = float(func(x))
            except Exception:
                fx = float('inf')
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Sobol sequence generator (simplified Halton/LCG mix for uniform coverage without scipy dependency issues if any, 
        # but standard library + numpy is fine. We can use a simple deterministic quasi-random or just well-spaced randoms.
        # The directive suggests Sobol. SciPy is available.
        from scipy.stats import qmc

        # Generate a Sobol sequence for initial exploration
        # We want to cover [-5, 5]^5
        # Let's generate 20 points for initial global search
        sobol_samples = qmc.Sobol(d=dim, scramble=True)
        num_sobol = 20
        sobol_points = sobol_samples.random_base2(n=num_sobol)
        sobol_points = sobol_points * 10.0 - 5.0 # Map [0,1] to [-5,5]

        # Phase 1: Global Exploration with Sobol points
        # Evaluate all Sobol points first to find a good basin
        for p in sobol_points:
            if calls_made >= total_budget or time.time() > deadline:
                break
            wrapper(p)

        # Phase 2: Local Exploitation with (1+1)-ES with adaptive sigma
        # Start from the best point found in Phase 1
        start_x = best_x.copy()
        
        # (1+1)-ES with 1/5 success rule for sigma adaptation
        def es_1plus1_adaptive(start_x, budget_limit):
            nonlocal calls_made, best_f, best_x
            
            x = start_x.copy()
            # We already evaluated start_x in phase 1 or it's the initial best
            # But we need fx for the current x to compare against
            # Since wrapper updates best_f, we need to know the fx of x.
            # If x is best_x, fx is best_f.
            # If x is not best_x, we need to evaluate it.
            # To be safe, let's evaluate the start point if it wasn't the global best at that moment.
            # Actually, best_x is the global best. So if we start from best_x, fx = best_f.
            fx = wrapper(x) # This ensures fx is correct and updates best if needed (though it shouldn't change)
            
            sigma = 0.5 # Initial step size
            successes = []
            window_size = 10
            steps = 0
            max_steps = budget_limit
            
            while steps < max_steps and calls_made < total_budget and time.time() < deadline:
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5, 5)
                
                fx_trial = wrapper(x_trial)
                
                # Accept or Reject
                if fx_trial < fx:
                    x = x_trial
                    fx = fx_trial
                    successes.append(1)
                else:
                    successes.append(0)
                    
                # Adapt sigma based on 1/5 success rule
                if len(successes) >= window_size:
                    success_rate = np.mean(successes[-window_size:])
                    if success_rate > 0.2:
                        sigma *= 1.05 # Increase
                    elif success_rate < 0.2:
                        sigma /= 1.05 # Decrease
                    # Cap sigma
                    sigma = np.clip(sigma, 1e-6, 5.0)
                    # Reset successes window to avoid long-term bias
                    successes = successes[-window_size:]
                    
                steps += 1
                
                # Stagnation check: if no improvement in best_f for last 30 steps, break
                # This is a simple heuristic. We can track best_f changes.
                # For now, rely on sigma shrinking to slow down, or just let it run until budget.
                # Given the budget is 500, and we used ~20 for Sobol, we have ~480 left.
                # (1+1)-ES is fast.

        # Run the adaptive ES
        # Use remaining budget
        remaining_budget = total_budget - calls_made
        es_1plus1_adaptive(start_x, remaining_budget)

        return best_x, best_f
