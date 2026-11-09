# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive sigma and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 275  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Global best
        best_x = None
        best_f = np.inf
        calls = 0

        # Phase 1: Few random samples to find a good starting point
        n_random = 5
        for i in range(n_random):
            if calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Should not happen if budget >= 1
            return None, np.inf

        # Phase 2: (1+1)-ES with restarts
        x = best_x.copy()
        fx = best_f
        sigma = 2.0  # Start with moderate step size
        successes = 0
        attempts = 0
        steps_since_improvement = 0

        while calls < budget and time.time() < deadline:
            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                # Accept
                x, fx = y, fy
                successes += 1
                steps_since_improvement = 0
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Reject
                steps_since_improvement += 1

            attempts += 1

            # Adapt sigma using 1/5 success rule
            if attempts >= 10:
                success_rate = successes / attempts
                if success_rate > 1.0 / 5.0:
                    sigma *= 1.2
                elif success_rate < 1.0 / 5.0:
                    sigma *= 0.8
                # Reset counters
                successes = 0
                attempts = 0

            # Restart mechanism:
            # If sigma is too small or we've been stuck for too long, restart
            if sigma < 1e-3 or steps_since_improvement > 100:
                if calls >= budget - 1:
                    break
                # Restart from best known point with a larger sigma
                x = best_x + rng.uniform(-0.5, 0.5, dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma = 1.0
                steps_since_improvement = 0
                successes = 0
                attempts = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
