    import numpy as np
    import time

    class Optimizer:
        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            start_time = time.time()
            deadline = start_time + 280 # Safety margin
            d = self.dim
            budget = self.budget
            
            # Budget allocation: 90% Global, 10% Local (min 10 for NM setup)
            local_budget = max(10, int(budget * 0.1))
            global_budget = budget - local_budget

            best_x = None
            best_f = float('inf')
            calls = 0

            def eval_point(x):
                nonlocal calls, best_x, best_f
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx

            # --- Differential Evolution ---
            NP = max(10, 5 * d) # Population size
            # Initialize population
            population = [np.random.uniform(-5, 5, d) for _ in range(NP)]
            for x in population:
                if calls >= global_budget: break
                eval_point(x)
            
            # DE Parameters
            CR = 0.9
            F = 0.8

            while calls < global_budget:
                if time.time() >= deadline:
                    break
                
                # Sort population to keep best at top
                # This is O(NP log NP), cheap for NP=25
                population.sort(key=lambda x: func(x) if False else 0) # Need to store fvals
                # Better: keep list of (f, x)
                
                # Let's refactor to keep fvals
                # ...
            