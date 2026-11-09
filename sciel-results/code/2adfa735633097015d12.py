"""Combined optimizer: structured global start + single high-budget Nelder-Mead, with (1+1)-ES fallback."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Strategy for BBOB f13 (Sharp Ridge):
    1. Structured global search: evaluate a set of diverse candidates (center, random, axis-aligned)
       to find a good starting point for local refinement.
    2. Single high-budget Nelder-Mead run from the best candidate to converge on the sharp ridge.
    3. If Nelder-Mead fails or uses too few calls, fall back to adaptive (1+1)-ES with restarts.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            calls_made[0] += 1
            if calls_made[0] > budget:
                raise Exception("Budget exceeded")

            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)

            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Structured global search to find a good starting point
        # Evaluate a diverse set of candidates: center, random points, axis-aligned points
        candidates = [np.zeros(n)]

        # Random points in [-4, 4] (where the minimizer lies)
        for _ in range(10):
            candidates.append(rng.uniform(-4, 4, n))

        # Axis-aligned points: vary one coordinate at a time
        for i in range(n):
            for val in [-3, -1, 1, 3]:
                cand = np.zeros(n)
                cand[i] = val
                candidates.append(cand)

        # Evaluate all candidates
        for cand in candidates:
            if calls_made[0] >= budget:
                break
            wrapped_func(cand)

        # Phase 2: Single high-budget Nelder-Mead from the best point
        # Allocate most of the remaining budget to Nelder-Mead
        nm_budget = max(10, budget - calls_made[0] - 50)  # Reserve 50 for fallback

        try:
            # Use Nelder-Mead with a large maxiter
            # The 'fun' callback will count calls via wrapped_func
            result = minimize(
                wrapped_func,
                best_x.copy(),
                method='Nelder-Mead',
                options={'maxiter': nm_budget, 'xatol': 1e-6, 'fatol': 1e-6}
            )

            # Update best if Nelder-Mead found something better
            if result.fun < best_f:
                best_f = result.fun
                best_x = np.clip(result.x, -5, 5)

        except Exception:
            pass

        # Phase 3: If we have budget left and haven't converged well, use (1+1)-ES fallback
        if calls_made[0] < budget:
            # Adaptive (1+1)-ES with 1/5 rule and stall detection
            window_size = 5
            successes = 0
            steps_in_window = 0
            stall_threshold = 30
            max_sigma = 5.0
            min_sigma = 1e-4

            x = best_x.copy()
            fx = best_f
            sigma = 1.0
            steps_without_improvement = 0

            while calls_made[0] < budget:
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = wrapped_func(y)

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    steps_without_improvement = 0
                    sigma = min(sigma * 1.2, max_sigma)
                else:
                    steps_without_improvement += 1
                    sigma = max(sigma * 0.85, min_sigma)

                steps_in_window += 1

                if steps_in_window >= window_size:
                    success_rate = successes / steps_in_window
                    if success_rate > 1/5:
                        sigma = min(sigma * 1.1, max_sigma)
                    else:
                        sigma = max(sigma * 0.9, min_sigma)
                    successes = 0
                    steps_in_window = 0

                if steps_without_improvement > stall_threshold:
                    # Restart from a random point near the best
                    x = best_x + rng.normal(0, 0.5, n)
                    x = np.clip(x, -5, 5)
                    fx = wrapped_func(x)
                    sigma = 1.0
                    steps_without_improvement = 0
                    successes = 0
                    steps_in_window = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
