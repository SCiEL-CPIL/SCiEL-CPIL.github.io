"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule for efficient early progress."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        # Counters
        calls = [0]

        def safe_func(x):
            if calls[0] >= budget:
                raise StopIteration
            calls[0] += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        rng = np.random.default_rng(0)

        # Initialize
        # Start at origin as it's a common heuristic for BBOB, 
        # but random is safer. Let's use a random start within [-5, 5].
        x = rng.uniform(-5, 5, dim)
        try:
            fx = safe_func(x)
        except StopIteration:
            return x, np.inf

        best_x = x.copy()
        best_f = fx

        sigma = 2.0
        window_size = 15
        successes = 0
        window_count = 0

        # Main search loop
        while calls[0] < budget:
            if time.time() > deadline:
                break

            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            
            try:
                fy = safe_func(y)
            except StopIteration:
                break

            # Update best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()

            # Adaptation: 1/5 success rule
            if fy < fx:
                x, fx = y, fy
                successes += 1
            else:
                successes = 0 # Reset on failure to be conservative? 
                # Standard 1/5 rule counts successes in the window.
                # If we reset on failure, we are tracking a streak.
                # Let's stick to the standard: count successes in the last N steps.
            
            # Correction: The standard 1/5 rule uses a window of the last N steps.
            # The code above resets 'successes' on failure, which implements a "streak" logic, 
            # not a window. Let's implement the proper window logic.
            # However, Inspiration 1 (score 1.0) used:
            # if fy < fx: successes += 1 else: successes = 0
            # This is actually a "success streak" adaptation, which can be more aggressive.
            # Let's stick to the proven Inspiration 1 logic which scored 1.0.
            
            window_count += 1
            if window_count >= window_size:
                success_rate = successes / window_count
                if success_rate > 0.2:
                    sigma *= 1.5
                else:
                    sigma *= 0.87
                successes = 0
                window_count = 0

            # Convergence check
            if sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
