"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid Multi-Start (1+1)-ES Optimizer with restarts and polish phase.
    
    Strategy:
    1. Initial Diversification: Sample diverse points to identify promising basins.
    2. Exploitation: Run (1+1)-ES with adaptive step size and restarts on stall.
    3. Polish: When sigma is very small, do a final high-precision pass.
    4. Safety: Strict deadline and budget enforcement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget
        calls_made = 0

        best_x = None
        best_f = np.inf

        # Phase 1: Initial Diversification
        n_init_samples = 12
        init_candidates = []
        
        # Strategic points for multimodal functions
        init_candidates.append(np.zeros(n))
        init_candidates.append(np.ones(n) * 2.5)
        init_candidates.append(np.ones(n) * -2.5)
        init_candidates.append(np.ones(n) * 1.5)
        init_candidates.append(np.ones(n) * -1.5)
        init_candidates.append(np.ones(n) * 3.5)
        init_candidates.append(np.ones(n) * -3.5)
        
        # Random points with some structure
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
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made = 1

        # Phase 2: (1+1)-ES with restarts
        x = best_x.copy()
        fx = best_f
        
        sigma = 2.0
        sigma_inc = 1.3
        sigma_dec = 0.85
        sigma_min = 1e-6
        sigma_max = 10.0

        max_restarts = 6
        restart_count = 0
        no_improve_counter = 0
        stall_threshold = 25

        remaining_budget = total_budget - calls_made

        for _ in range(remaining_budget):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break

            # Generate offspring
            noise = rng.standard_normal(n)
            y = x + sigma * noise
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= sigma_inc
                no_improve_counter = 0
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= sigma_dec
                no_improve_counter += 1

            # Clamp sigma
            sigma = max(sigma_min, min(sigma, sigma_max))

            # Restart logic: if stuck for too long, restart from best with new sigma
            if no_improve_counter >= stall_threshold and restart_count < max_restarts:
                restart_count += 1
                no_improve_counter = 0
                # Restart from best point with a moderate step size
                x = best_x.copy()
                fx = best_f
                sigma = 1.5 * (0.8 ** restart_count)  # Decreasing restart sigma
                sigma = max(sigma, 0.5)

            # Polish phase: if sigma is very small, do fine-tuning
            if sigma < 1e-4 and calls_made < total_budget - 5:
                # Do a few very fine steps
                for _ in range(min(5, total_budget - calls_made)):
                    if time.time() > deadline:
                        break
                    noise = rng.standard_normal(n) * 1e-4
                    y = x + noise
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
