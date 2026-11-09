"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive multi-restart optimizer for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        rng = np.random.default_rng(0)
        d = self.dim
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def eval_func(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)
        
        # Strategy: Adaptive (1+1)-ES with restarts and diverse initializations
        # For BBOB f4 (Büche-Rastrigin), which has many local minima,
        # we use multiple restarts with different scales and starting points.
        
        def run_es(x0, sigma0, max_calls):
            nonlocal best_x, best_f, calls_made
            
            x = x0.copy()
            fx = eval_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = sigma0
            success_count = 0
            last_improvement = 0
            
            while calls_made < self.budget:
                if time.time() > deadline - 5.0:
                    break
                
                # Generate trial point
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = eval_func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                    success_count += 1
                    last_improvement = calls_made
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                
                # Restart if no improvement for a while
                if calls_made - last_improvement > 50 and sigma < 0.01:
                    break
            
            return x, fx
        
        # Phase 1: Coarse search with large steps
        # Try several starting points
        starts = [
            np.zeros(d),
            rng.uniform(-2, 2, d),
            rng.uniform(-3, 3, d),
        ]
        
        for i, x0 in enumerate(starts):
            if time.time() > deadline - 30.0:
                break
            if calls_made >= self.budget:
                break
            
            # Use a fraction of the budget for each coarse phase
            budget_alloc = min(self.budget - calls_made, 150)
            
            # Temporarily adjust budget
            old_budget = self.budget
            self.budget = calls_made + budget_alloc
            
            run_es(x0, 2.0, budget_alloc)
            
            self.budget = old_budget
        
        # Phase 2: Fine-tuning from best found so far
        if best_x is not None and calls_made < self.budget:
            if time.time() < deadline - 10.0:
                run_es(best_x.copy(), 0.5, self.budget - calls_made)
        
        # Phase 3: If budget remains, try another exploration
        if calls_made < self.budget and time.time() < deadline - 10.0:
            x0 = rng.uniform(-4, 4, d)
            run_es(x0, 1.0, self.budget - calls_made)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
