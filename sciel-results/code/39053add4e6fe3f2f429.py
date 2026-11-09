"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # 300s limit, keep 30s margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_time():
            return (time.time() - start_time) < deadline

        # Generate diverse starting points
        # Mix of random points and points near grid
        n_starts = 15
        starts = []
        # Some random
        for i in range(n_starts):
            if i < 5:
                # Random uniform
                x = rng.uniform(-4, 4, dim)
            elif i < 10:
                # Near grid points (Rastrigin has local minima near integers)
                x = rng.uniform(-4, 4, dim)
                # Round some coordinates to nearest integer
                for d in range(dim):
                    if rng.random() < 0.5:
                        x[d] = round(x[d])
            else:
                # Random with bias toward center
                x = rng.uniform(-3, 3, dim)
            starts.append(x)

        # Evaluate all starts
        for x in starts:
            if calls_made >= budget or not check_time():
                break
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Run (1+1)-ES from the best start, with restarts if stagnating
        # We'll do multiple short runs with restarts
        remaining = budget - calls_made
        if remaining <= 0 or not check_time():
            return best_x, best_f

        # Strategy: run ES from best point, if stagnating for a while, restart from a new random point
        # Keep track of best found so far
        
        # Parameters
        sigma0 = 1.5
        stagnation_limit = 30  # if no improvement for 30 steps, consider restart
        max_es_iter_per_run = 40  # max iterations per ES run before forced restart

        # We'll do a loop of ES runs
        total_es_calls = 0
        es_calls_budget = remaining

        # Start from best
        x = best_x.copy()
        fx = best_f
        sigma = sigma0

        # Adaptive step size with 1/5 rule
        # Track success rate over a window
        window_size = 15
        successes = 0
        window_count = 0

        # Also track stagnation
        last_improve_call = 0
        current_best_in_es = fx

        while total_es_calls < es_calls_budget and check_time():
            # Check stagnation - if no improvement for stagnation_limit calls, restart
            if (total_es_calls - last_improve_call) >= stagnation_limit and (total_es_calls - last_improve_call) >= max_es_iter_per_run:
                # Restart from a new random point
                x_new = rng.uniform(-4, 4, dim)
                fx_new = func(x_new)
                total_es_calls += 1
                calls_made += 1
                if fx_new < best_f:
                    best_f = fx_new
                    best_x = x_new.copy()
                    current_best_in_es = fx_new
                    last_improve_call = total_es_calls
                x = x_new
                fx = fx_new
                sigma = sigma0
                successes = 0
                window_count = 0
                continue

            # Generate offspring
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            total_es_calls += 1
            calls_made += 1

            if fy < fx:
                x = y
                fx = fy
                successes += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    last_improve_call = total_es_calls
                if fx < current_best_in_es:
                    current_best_in_es = fx

            window_count += 1
            if window_count >= window_size:
                # Adapt sigma based on success rate
                success_rate = successes / window_count
                if success_rate > 0.3:
                    sigma *= 1.2
                elif success_rate < 0.1:
                    sigma *= 0.7
                else:
                    sigma *= 1.0
                # Clamp sigma
                sigma = np.clip(sigma, 0.05, 5.0)
                successes = 0
                window_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
