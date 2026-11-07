import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        calls = 0
        best_x = None
        best_f = float('inf')
        
        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Initial Exploration
        # Sample a diverse set of points to identify promising basins.
        # Rastrigin has many local minima. We want to find the global one or a very good local one.
        n_init = 30
        candidates = []
        
        # Always try the origin and center of the box, as shifts are often small or zero
        eval_func(np.zeros(dim))
        
        for _ in range(n_init - 1):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = eval_func(x)
            candidates.append((f, x))
            
        # Sort candidates by function value
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Multi-start Local Optimization
        # For the top K candidates, run a (1+1)-ES to refine them.
        # This combines the "structural" idea of exploring multiple basins (from current program's intent)
        # with the "refinement machinery" of the (1+1)-ES (from inspiration 1).
        
        top_k = 3
        # Allocate budget: reserve some for final polish, distribute rest among top_k
        # Let's say 60% of remaining budget for multi-start, 40% for final polish
        # But we need to be careful not to exceed total budget.
        
        # Let's just run ES for each top candidate until stagnation or a fixed local budget.
        # A fixed local budget is safer for time management.
        local_budget_per_candidate = 100
        
        for i in range(top_k):
            if calls >= budget or time.time() > deadline:
                break
            if i >= len(candidates):
                break
            
            f_start, x_start = candidates[i]
            
            # If the starting point is already very good, we might not need to spend much here?
            # But we don't know f_opt. Let's just run the ES.
            
            x = x_start.copy()
            f_x = f_start
            
            # (1+1)-ES Parameters
            sigma = 0.5  # Initial step size. Rastrigin minima are spaced by ~1.0, so 0.5 is reasonable to find neighbors or refine.
            window = 10
            successes = 0
            stagnation = 0
            max_stagnation = 30
            
            local_calls = 0
            while calls < budget and time.time() < deadline and local_calls < local_budget_per_candidate:
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                f_y = eval_func(y)
                local_calls += 1
                
                if f_y < f_x:
                    x = y
                    f_x = f_y
                    successes += 1
                    stagnation = 0
                else:
                    stagnation += 1
                    
                # Adapt sigma based on success rate
                if successes % window == 0:
                    sr = successes / window
                    if sr > 0.5:
                        sigma *= 1.2
                    elif sr < 0.3:
                        sigma *= 0.8
                    else:
                        sigma *= 1.0
                    sigma = np.clip(sigma, 0.01, 5.0)
                    successes = 0
                    
                # Stagnation check
                if stagnation > max_stagnation:
                    break

        # Phase 3: Final Polish
        # Use remaining budget to refine the global best found so far.
        # Use a very small sigma to converge to the minimum.
        if calls < budget and time.time() < deadline and best_x is not None:
            x = best_x.copy()
            f_x = best_f
            sigma = 0.05
            
            # Run until budget or time runs out, or stagnation
            stagnation = 0
            max_stagnation = 50
            
            while calls < budget and time.time() < deadline:
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                f_y = eval_func(y)
                
                if f_y < f_x:
                    x = y
                    f_x = f_y
                    stagnation = 0
                    # Slowly decrease sigma as we get closer?
                    # Or keep it constant if we are just polishing?
                    # Let's decrease it slightly to ensure convergence.
                    sigma *= 0.95
                    sigma = max(sigma, 0.001)
                else:
                    stagnation += 1
                    # If stagnant, maybe increase sigma slightly to escape?
                    # But this is "polish", so we want to stay close.
                    # Let's just break if stagnant for too long.
                    if stagnation > max_stagnation:
                        break

        return best_x, best_f
