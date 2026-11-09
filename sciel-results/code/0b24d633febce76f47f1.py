"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid Multi-Start (1+1)-ES Optimizer with Restart Mechanism.
    
    Strategy:
    1. Initial Diversification: Sample diverse points (origin, structured mid-range, random)
       to identify a promising basin. This is crucial for multimodal functions 
       like Rastrigin and Gallagher where the global minimum may not be near the origin.
    2. Exploitation: Run an aggressive (1+1)-Evolution Strategy from the best 
       initial point. This algorithm has minimal overhead per iteration, allowing 
       for rapid step-size adaptation and frequent updates, which is highly 
       beneficial for the AOCC score that rewards early progress.
    3. Restart Mechanism: When the search stalls (no improvement for a threshold number
       of iterations), restart from a perturbed version of the best solution or from
       a new random/structured point to escape local minima.
    4. Safety: Strict deadline and budget enforcement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s. We stop at 290s to be safe.
        deadline = start_time + 290.0 
        
        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget
        calls_made = 0

        best_x = None
        best_f = np.inf

        # Phase 1: Initial Diversification
        # We use ~10-15% of the budget for exploration to ensure we don't get stuck 
        # in a poor local minimum, which is common in Rastrigin/Gallagher.
        n_init_samples = 12
        init_candidates = []
        
        # 1. Origin
        init_candidates.append(np.zeros(n))
        
        # 2. Structured mid-range points (helps if the shift is moderate)
        init_candidates.append(np.ones(n) * 2.0)
        init_candidates.append(np.ones(n) * -2.0)
        init_candidates.append(np.ones(n) * 1.0)
        init_candidates.append(np.ones(n) * -1.0)
        init_candidates.append(np.ones(n) * 3.0)
        init_candidates.append(np.ones(n) * -3.0)
        
        # 3. Random points to cover the rest of the space
        for _ in range(n_init_samples - 7):
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

        # Phase 2: Aggressive (1+1)-ES from the best initial point with restarts
        x = best_x.copy()
        fx = best_f
        
        # Initial step size: larger to explore more aggressively
        sigma = 2.0
        
        # Parameters for adaptation
        sigma_inc = 1.3
        sigma_dec = 0.85
        sigma_min = 1e-4
        sigma_max = 10.0

        max_restarts = 5
        restart_count = 0
        no_improve_counter = 0
        stall_threshold = 25

        # Main Optimization Loop
        while calls_made < total_budget:
            if time.time() > deadline:
                break

            # Generate offspring
            noise = rng.standard_normal(n)
            y = x + sigma * noise
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
                    no_improve_counter = 0
            else:
                # Failure: stay at x, decrease step size
                sigma *= sigma_dec
                no_improve_counter += 1

            # Clamp sigma to valid range
            sigma = max(sigma_min, min(sigma, sigma_max))

            # Check for stall and restart if necessary
            if no_improve_counter > stall_threshold and restart_count < max_restarts:
                restart_count += 1
                no_improve_counter = 0
                
                # Restart from a perturbed version of the best solution
                # or from a new structured/random point
                if restart_count <= 2:
                    # Perturb the current best solution with moderate noise
                    x = best_x + 1.5 * rng.standard_normal(n)
                    x = np.clip(x, -5, 5)
                elif restart_count <= 4:
                    # Try structured points near the best
                    offset = 2.0 * rng.choice([-1, 1], size=n)
                    x = best_x + offset
                    x = np.clip(x, -5, 5)
                else:
                    # Try a completely new random point
                    x = rng.uniform(-5, 5, n)
                
                fx = func(x)
                calls_made += 1
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                # Reset sigma for fresh exploration
                sigma = 2.0

        return best_x, best_f
# EVOLVE-BLOCK-END
