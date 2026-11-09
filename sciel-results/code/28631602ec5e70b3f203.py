import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        n = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse grid search to quickly find a good region
        # Use a structured grid with 3 points per dimension (27 total for 5D)
        grid_points = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        for m in range(-1, 2):
                            x = np.array([i * 3.0, j * 3.0, k * 3.0, l * 3.0, m * 3.0])
                            x = np.clip(x, -5, 5)
                            grid_points.append(x)
        
        # Evaluate grid points (stop early if budget is tight)
        for x in grid_points:
            if calls_made >= total_budget or time.time() > deadline:
                break
            evaluate(x)
        
        # Phase 2: Deterministic multi-scale local search with geometric step decay
        if best_x is not None and calls_made < total_budget:
            # Fixed sequence of step sizes (geometric decay)
            step_sizes = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625]
            
            # For each step size, try moves in all coordinate directions (+ and -)
            for step in step_sizes:
                if calls_made >= total_budget or time.time() > deadline:
                    break
                
                improved = False
                for dim_idx in range(n):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    
                    # Try positive direction
                    x_pos = best_x.copy()
                    x_pos[dim_idx] += step
                    x_pos = np.clip(x_pos, -5, 5)
                    f_pos = evaluate(x_pos)
                    if f_pos < best_f:
                        improved = True
                        continue  # best_x already updated in evaluate
                    
                    # Try negative direction
                    x_neg = best_x.copy()
                    x_neg[dim_idx] -= step
                    x_neg = np.clip(x_neg, -5, 5)
                    f_neg = evaluate(x_neg)
                    if f_neg < best_f:
                        improved = True
                
                # If no improvement at this scale, we could break early,
                # but continue to smaller scales as the function might have local structure
                if not improved and step < 0.01:
                    # If we're at very small scales and no improvement, stop
                    break
        
        # Phase 3: Final random perturbation to escape potential local minima
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            rng = np.random.default_rng(42)
            remaining = total_budget - calls_made
            n_random = min(remaining, 20)
            
            for _ in range(n_random):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                x = best_x + 0.1 * rng.standard_normal(n)
                x = np.clip(x, -5, 5)
                evaluate(x)
        
        # Fallback
        if best_x is None:
            x = np.zeros(n)
            best_f = func(x)
            best_x = x

        return best_x, best_f
