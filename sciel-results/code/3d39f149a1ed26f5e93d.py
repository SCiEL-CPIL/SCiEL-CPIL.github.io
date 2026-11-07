import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 290.0
        
        budget = self.budget
        dim = self.dim
        calls = 0
        best_f = float('inf')
        best_x = None
        
        # Helper to evaluate and track
        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            val = func(x)
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val
        
        # Phase 1: Global exploration
        n_explore = min(30, budget)
        for i in range(n_explore):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)
        
        if best_x is None:
            return None, None
        
        # Phase 2: Multi-start (1+1)-ES with per-dimension step sizes
        # Use the best few points from exploration as starting points
        # We'll do sequential ES runs from the best points
        
        # Collect top points from exploration (we need to re-evaluate or keep track)
        # Since we didn't keep all points, let's just use the current best and a few random perturbations
        # Actually, a better approach: do a single long (1+1)-ES from the best point,
        # but with adaptive step sizes per dimension.
        
        # For sharp ridge, we need to align with the ridge direction.
        # (1+1)-ES with per-dimension step sizes can adapt to this.
        
        # Strategy:
        # 1. Start from best_x
        # 2. Use per-dimension step sizes initialized to 1.0
        # 3. Adapt step sizes based on success/failure
        # 4. If stagnation, restart from a random point or a perturbation of best
        
        # Let's implement a robust (1+1)-ES:
        
        def run_es(x_start, n_evals):
            nonlocal calls, best_f, best_x
            x = x_start.copy()
            f = evaluate(x)
            if f is None:
                return x, best_f
            
            # Per-dimension step sizes
            sigma = np.ones(dim)
            # Initial sigma based on problem scale
            sigma *= 2.0  # Start with relatively large steps
            
            # Adaptation rates
            c_sigma = 1.0 / (4 * dim)  # Learning rate for sigma
            c_succ = 0.8  # Success rate target
            
            # Success counter
            succ = 0.0
            window = dim + 1  # Window size for success rate
            
            for i in range(n_evals):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Generate mutant
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                
                # Clip to bounds
                x_trial = np.clip(x_trial, -5, 5)
                
                f_trial = evaluate(x_trial)
                if f_trial is None:
                    break
                
                # Update success counter
                if f_trial < f:
                    succ += 1
                    x = x_trial
                    f = f_trial
                else:
                    succ -= 1
                
                # Adapt sigma
                # Simple adaptation: increase if success rate > 0.5, decrease otherwise
                # Using a cumulative success rate
                if i > 0:
                    # Update sigma based on recent success
                    # Use a simple rule: if last evaluation was successful, increase sigma slightly
                    # Otherwise decrease
                    if f_trial < f:
                        # Success: increase sigma
                        sigma *= 1.05
                    else:
                        # Failure: decrease sigma
                        sigma *= 0.95
                
                # Clip sigma to reasonable range
                sigma = np.clip(sigma, 1e-8, 10.0)
            
            return x, f
        
        # Run ES from the best point
        # Use most of the remaining budget
        remaining = budget - calls
        n_es = max(1, int(remaining * 0.8))
        
        x, f = run_es(best_x, n_es)
        
        # If there's still budget, do a few more restarts
        remaining = budget - calls
        if remaining > 20:
            # Do a few short ES runs from random points
            n_restarts = min(3, remaining // 10)
            for r in range(n_restarts):
                if calls >= budget or time.time() > deadline:
                    break
                x_rand = rng.uniform(-5, 5, dim)
                n_es_short = max(1, (budget - calls) // (n_restarts - r + 1))
                x, f = run_es(x_rand, n_es_short)
        
        return best_x, best_f
