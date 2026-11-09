"""Rewritten optimizer: restart-based (1+1)-ES with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Restart-based (1+1)-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Phase 1: Quick random sampling to find a good starting point
        n_init = 20
        for i in range(n_init):
            if calls_made >= self.budget:
                break
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback: center
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls_made += 1

        # Phase 2: Restart-based (1+1)-ES
        # Use remaining budget for ES iterations
        remaining_budget = self.budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f

        # ES parameters
        sigma = 1.5  # initial step size
        success_threshold = 1/5  # 1/5 success rule
        successes = 0
        failures = 0
        window = 5
        restart_threshold = self.dim * 1.5  # restart if no improvement for this many steps

        # We'll do multiple restarts
        max_restarts = 8
        restarts_done = 0

        for restart in range(max_restarts):
            if calls_made >= self.budget:
                break
            if time.time() > deadline:
                break

            # Start from best point found so far, or a new random point for later restarts
            if restart == 0:
                x = best_x.copy()
                fx = best_f
            else:
                # Perturb from best or sample new
                if restarts_done < 3:
                    # Small perturbation from best
                    x = np.clip(best_x + 0.5 * rng.standard_normal(self.dim), -5, 5)
                else:
                    # Larger perturbation or new random
                    x = np.clip(best_x + 2.0 * rng.standard_normal(self.dim), -5, 5)
                    # Clip to box
                    x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            # Reset per-restart counters
            no_improve_count = 0
            successes = 0
            failures = 0
            sigma = 1.5 if restart == 0 else 1.0
            steps_in_restart = 0
            max_steps_per_restart = max(10, remaining_budget // (max_restarts - restart))

            for step in range(max_steps_per_restart):
                if calls_made >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    no_improve_count = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Increase sigma on success
                    sigma *= 1.25
                    # Cap sigma
                    if sigma > 5.0:
                        sigma = 5.0
                else:
                    failures += 1
                    no_improve_count += 1
                    # Decrease sigma on failure
                    sigma *= 0.85
                    # Floor sigma
                    if sigma < 0.01:
                        sigma = 0.01

                # 1/5 rule: adjust sigma based on success rate over window
                total = successes + failures
                if total >= window:
                    rate = successes / total
                    if rate > 0.2:
                        sigma *= 1.1
                    elif rate < 0.1:
                        sigma *= 0.9
                    # Reset window counters
                    successes = 0
                    failures = 0

                steps_in_restart += 1

                # Early restart if stuck
                if no_improve_count >= restart_threshold and step > 5:
                    restarts_done += 1
                    break

            restarts_done += 1

            # If we've used most of our budget, stop
            if calls_made >= self.budget - 5:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
