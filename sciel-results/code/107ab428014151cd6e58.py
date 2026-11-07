import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 270.0
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        calls = [0]
        best_x = np.zeros(dim)
        best_f = float('inf')

        def evaluate(x):
            nonlocal best_x, best_f
            if calls[0] >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls[0] += 1
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global Exploration using Sobol Sequence
        # 100 points in 5 dimensions provides good low-discrepancy coverage.
        n_global = 100
        try:
            from scipy.stats.qmc import Sobol
            sampler = Sobol(d=dim, scramble=True, seed=42)
            global_points = sampler.random(n_global)
            global_points = global_points * 10.0 - 5.0
        except ImportError:
            global_points = rng.uniform(-5, 5, size=(n_global, dim))

        for x in global_points:
            if calls[0] >= budget or time.time() > deadline:
                break
            evaluate(x)

        if calls[0] == 0:
            evaluate(np.zeros(dim))
            return best_x, best_f

        # Phase 2: Local Refinement using (1+1)-ES with Momentum and Restart Logic
        # We will run multiple (1+1)-ES instances. 
        # Each instance starts from a candidate point (initially the global best, then random restarts).
        # The (1+1)-ES uses a simple step-size adaptation (1/5 rule variant) and momentum.
        # We implement a "basin-aware" restart: if stagnation is detected, we jump to a new random point
        # or a point near the current best with a larger step size to escape local minima.

        remaining_budget = budget - calls[0]
        if remaining_budget <= 0:
            return best_x, best_f

        # We will divide the remaining budget into chunks for multiple ES runs.
        # Let's say we allow up to 5 restarts/ES runs.
        # Each run gets a portion of the budget.
        # To maximize early progress, we run the first one (from global best) for a significant portion,
        # and subsequent ones for shorter periods if the first one gets stuck.

        # However, a single long (1+1)-ES with restarts is often more efficient than multiple short ones
        # because it retains momentum and step-size adaptation state.
        # Let's implement a single loop that acts as a (1+1)-ES but allows "hard restarts" 
        # to new random points if stagnation persists, effectively doing multi-start within one loop.

        # State variables for the ES
        # Start from the best point found in Phase 1
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # Initial step size
        momentum = np.zeros(dim)
        
        # 1/5 rule parameters
        success = 0.0
        # We track no-improvement count for restart logic
        no_improve_count = 0
        restart_threshold = 50  # If no improvement for 50 steps, consider a restart
        
        # We will run until budget or deadline
        while calls[0] < budget and time.time() < deadline:
            # Generate candidate
            y = x + sigma * (rng.standard_normal(dim) + momentum)
            y = np.clip(y, -5.0, 5.0)
            fy = evaluate(y)
            
            if fy < fx:
                # Success
                x = y
                fx = fy
                sigma *= 1.1  # Expand step size
                success = 1.0
                no_improve_count = 0
                # Update momentum towards the successful direction
                # Simple momentum: average of recent directions? 
                # For simplicity, let's just keep momentum small or reset it on success?
                # Actually, standard (1+1) ES doesn't use momentum. 
                # Let's stick to pure (1+1) ES with step size adaptation for stability, 
                # but the "restart" mechanism handles the multi-modality.
            else:
                # Failure
                sigma *= 0.9  # Shrink step size
                success = 0.0
                no_improve_count += 1
            
            # Step size adaptation based on 1/5 rule (approximate)
            # If success rate is around 0.2, it's optimal.
            # We'll just use the multiplicative factors above which are standard heuristics.
            
            # Check for stagnation and potential restart
            if no_improve_count > restart_threshold:
                # If we are stuck, maybe we are in a local minimum.
                # Option 1: Shrink sigma further to try to find a tiny improvement (unlikely if Rastrigin).
                # Option 2: Jump to a new random location (Global Restart).
                # Option 3: Jump to a location near the current global best (Local Restart).
                
                # Let's try a jump to a new random point to explore other basins.
                # But we should keep the global best safe.
                # We will restart the local search from a new random point.
                # However, we must ensure we don't lose the global best.
                # The global best is tracked in best_x/best_f.
                
                # Let's pick a new random point.
                new_x = rng.uniform(-5.0, 5.0, dim)
                new_f = evaluate(new_x)
                
                if new_f < fx:
                    # New point is better than the current local point
                    x = new_x
                    fx = new_f
                else:
                    # New point is worse. 
                    # Should we go back to the previous local best or the global best?
                    # Let's go back to the global best to restart refinement from the best known point.
                    if best_x is not None:
                        x = best_x.copy()
                        fx = best_f
                    else:
                        x = new_x
                        fx = new_f
                
                # Reset step size for the new start
                sigma = 1.0
                no_improve_count = 0

        return best_x, best_f
