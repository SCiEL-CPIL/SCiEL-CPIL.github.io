"""Combined program: Multi-start (1+1)-ES with robust adaptation and refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with robust adaptation and refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin from 300s
        dim = self.dim
        budget = self.budget

        # Wrap func to track calls and time
        call_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        def run_11es(x0, sigma0, max_evals, window_size=10, min_sigma=1e-10, max_sigma=10.0):
            """Run (1+1)-ES with 1/5 success rule adaptation and occasional large jumps."""
            nonlocal call_count, best_f, best_x, deadline, budget
            
            x = x0.copy()
            fx = wrapped_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = sigma0
            successes = []
            no_improve_count = 0
            
            for step in range(max_evals):
                if time.time() >= deadline or call_count >= budget:
                    break
                
                # Occasionally use a larger step to escape local minima
                # Weierstrass is highly multimodal, so occasional large jumps help
                if rng.random() < 0.05:  # 5% chance of a large jump
                    step_dir = rng.standard_normal(dim)
                    step_size = sigma * 5.0
                else:
                    step_dir = rng.standard_normal(dim)
                    step_size = sigma
                
                candidate = np.clip(x + step_size * step_dir, -5, 5)
                f_candidate = wrapped_func(candidate)
                
                if f_candidate < fx:
                    # Success
                    x = candidate
                    fx = f_candidate
                    successes.append(1)
                    no_improve_count = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    successes.append(0)
                    no_improve_count += 1
                
                # Maintain window
                if len(successes) > window_size:
                    successes.pop(0)
                
                # Adapt sigma based on 1/5 success rule
                if len(successes) == window_size:
                    success_rate = sum(successes) / window_size
                    if success_rate > 0.2:
                        sigma *= 1.15
                    elif success_rate < 0.2:
                        sigma *= 0.85
                    # else keep sigma
                    
                    # Clamp sigma
                    sigma = np.clip(sigma, min_sigma, max_sigma)
                
                # Stop if no improvement for many iterations and sigma is small
                if no_improve_count > 30 and sigma < 0.01:
                    break

        # Phase 1: Exploration with multiple starts
        # Allocate 60% of budget to exploration
        alloc_explore = int(budget * 0.6)
        n_starts = 5
        starts = [np.zeros(dim)]
        for i in range(n_starts - 1):
            starts.append(rng.uniform(-3, 3, dim))
        
        # Distribute budget among starts
        steps_per_start = alloc_explore // n_starts
        
        for i, x0 in enumerate(starts):
            if time.time() >= deadline or call_count >= budget:
                break
            
            # Use remaining budget for this start, capped by steps_per_start
            remaining_for_start = min(steps_per_start, budget - call_count)
            if remaining_for_start < 10:
                break
                
            run_11es(x0, 1.0, remaining_for_start)

        # Phase 2: Exploitation on the best point found
        # Allocate remaining 40% (or whatever is left) to refinement
        if best_x is not None and call_count < budget:
            remaining = budget - call_count
            if remaining > 10:
                # Start refinement with smaller sigma
                run_11es(best_x, 0.05, remaining, window_size=8, min_sigma=1e-12)

        return best_x, best_f
# EVOLVE-BLOCK-END
