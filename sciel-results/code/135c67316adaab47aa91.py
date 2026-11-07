"""Portfolio-based optimizer for BBOB f3 (Rastrigin) with multi-phase local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Portfolio search:
    1. Brief exploratory phase with random sampling to find promising basins.
    2. Multi-phase local refinement using (1+1)-ES:
       - Phase 1: Polish the current best (small sigma).
       - Phase 2: Escape local minima (medium perturbation + search).
       - Phase 3: Re-explore wider area (large perturbation + search).
       - Phase 4: Final high-precision polish on the global best.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        # Global state
        best_x = None
        best_f = float('inf')
        calls_made = 0
        max_calls = self.budget
        dim = self.dim

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= max_calls

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Exploratory random search to find good starting points
        # Use a small fraction of the budget for this.
        n_explore = min(50, max(10, max_calls // 10))
        
        rng = np.random.default_rng(42)
        
        # Start with the center
        x0 = np.zeros(dim)
        evaluate(x0)
        
        # Add random points
        for _ in range(n_explore - 1):
            if check_deadline():
                break
            x = rng.uniform(-4.5, 4.5, dim)
            evaluate(x)

        if best_x is None:
            # Fallback
            x = np.zeros(dim)
            evaluate(x)
            return best_x, best_f

        # Define the local search function: (1+1)-ES
        def local_search(x_start, f_start, budget_left, sigma_init, rng_local):
            """
            Runs a (1+1)-ES local search from x_start.
            Updates global best_x and best_f if a better solution is found.
            """
            nonlocal calls_made, best_x, best_f
            
            x = x_start.copy()
            f = f_start
            sigma = sigma_init
            
            # 1/5 success rule variant: simple multiplicative adaptation
            # On success, expand slightly; on failure, contract.
            # For Rastrigin, which has periodic local minima, 
            # moderate expansion helps jump to adjacent minima if they are better (due to shift).
            
            for i in range(budget_left):
                if check_deadline():
                    break
                
                # Generate step
                step = sigma * rng_local.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                
                fy = func(y)
                calls_made += 1
                
                if fy < f:
                    # Accept
                    x = y
                    f = fy
                    sigma *= 1.1  # Expand
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    # Reject
                    sigma *= 0.9  # Contract
                
                # Clamp sigma to reasonable bounds
                sigma = np.clip(sigma, 0.005, 5.0)

            return x, f

        # Outer loop: Multi-phase refinement
        remaining = max_calls - calls_made
        if remaining <= 0:
            return best_x, best_f

        # We divide the remaining budget into 4 phases.
        # Phase 1: Polish (High precision, low noise)
        # Phase 2: Escape (Medium noise, jump to nearby basin)
        # Phase 3: Re-explore (High noise, jump to further basin)
        # Phase 4: Final Polish (Very high precision on the best found so far)
        
        # Allocation: 40% Polish, 25% Escape, 20% Re-explore, 15% Final Polish
        # This ensures we spend significant time refining the best candidate found in exploration.
        
        budget_polish = int(remaining * 0.40)
        budget_escape = int(remaining * 0.25)
        budget_reexplore = int(remaining * 0.20)
        budget_final = remaining - budget_polish - budget_escape - budget_reexplore

        # Ensure minimums if budget is very small
        if budget_polish < 5:
            budget_polish = remaining
            budget_escape = 0
            budget_reexplore = 0
            budget_final = 0

        # Phase 1: Polish current best
        if budget_polish > 0:
            local_search(best_x, best_f, budget_polish, sigma_init=0.5, rng_local=rng)
            remaining = max_calls - calls_made

        # Phase 2: Escape
        if budget_escape > 0 and not check_deadline():
            # Perturb best_x with moderate noise
            sigma_perturb = 1.0
            x_start2 = np.clip(best_x + sigma_perturb * rng.standard_normal(dim), -5, 5)
            f_start2 = func(x_start2)
            calls_made += 1
            if f_start2 < best_f:
                best_f = f_start2
                best_x = x_start2.copy()
            
            local_search(best_x, best_f, budget_escape, sigma_init=1.0, rng_local=rng)
            remaining = max_calls - calls_made

        # Phase 3: Re-explore
        if budget_reexplore > 0 and not check_deadline():
            # Perturb best_x with larger noise
            sigma_perturb = 2.0
            x_start3 = np.clip(best_x + sigma_perturb * rng.standard_normal(dim), -5, 5)
            f_start3 = func(x_start3)
            calls_made += 1
            if f_start3 < best_f:
                best_f = f_start3
                best_x = x_start3.copy()
            
            local_search(best_x, best_f, budget_reexplore, sigma_init=1.5, rng_local=rng)
            remaining = max_calls - calls_made

        # Phase 4: Final Polish
        if budget_final > 0 and not check_deadline():
            # Use a very small sigma for fine-tuning
            local_search(best_x, best_f, budget_final, sigma_init=0.1, rng_local=rng)

        return best_x, best_f

# EVOLVE-BLOCK-END
