"""Rewritten optimizer for BBOB-style Weierstrass f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with 1/5-rule adaptation and polish phase."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Multi-start: evaluate several diverse points to find a good basin
        candidates = [np.zeros(dim)]
        for _ in range(7):
            candidates.append(rng.uniform(-4, 4, dim))

        best_x = None
        best_f = np.inf
        for c in candidates:
            fc = func(c)
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        calls_made = len(candidates)

        # (1+1)-ES parameters
        sigma = 1.0
        success_threshold = 0.2  # target success rate for 1/5-rule
        # Track success over a sliding window
        window_size = 15
        successes = []

        # Phase 1: (1+1)-ES main search
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate trial point
            z = rng.standard_normal(dim)
            trial_x = best_x + sigma * z
            trial_x = np.clip(trial_x, -5, 5)

            trial_f = func(trial_x)
            calls_made += 1

            if trial_f < best_f:
                best_f = trial_f
                best_x = trial_x.copy()
                successes.append(1)
            else:
                successes.append(0)

            # Adapt sigma using 1/5-rule on sliding window
            if len(successes) >= window_size:
                recent_successes = successes[-window_size:]
                success_rate = sum(recent_successes) / window_size
                if success_rate > success_threshold:
                    sigma *= 1.05  # increase step
                elif success_rate < success_threshold:
                    sigma *= 0.95  # decrease step
                # Clamp sigma to reasonable range
                sigma = max(1e-8, min(sigma, 10.0))

        # Phase 2: Polish with smaller steps if we have budget left
        if calls_made < budget and best_f < 50:
            # Try a local polish with very small steps
            polish_sigma = sigma * 0.1
            polish_calls = 0
            max_polish = min(budget - calls_made, 50)
            for _ in range(max_polish):
                if time.time() > deadline or calls_made >= budget:
                    break
                z = rng.standard_normal(dim)
                trial_x = best_x + polish_sigma * z
                trial_x = np.clip(trial_x, -5, 5)
                trial_f = func(trial_x)
                calls_made += 1
                if trial_f < best_f:
                    best_f = trial_f
                    best_x = trial_x.copy()
                    polish_sigma *= 1.02
                else:
                    polish_sigma *= 0.98
                polish_sigma = max(1e-10, min(polish_sigma, 1.0))

        return best_x, best_f
# EVOLVE-BLOCK-END
