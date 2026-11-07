"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_exceeded():
            return time.time() > deadline

        def run_es(x0, sigma0, max_evals, rng, func):
            """Run (1+1)-ES from x0 with sigma0 for max_evals evaluations."""
            x = x0.copy()
            fx = func(x)
            sigma = sigma0
            best_x = x.copy()
            best_f = fx
            calls = 1

            for _ in range(max_evals - 1):
                if time_exceeded():
                    break
                if calls >= max_evals:
                    break

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                # Adaptive step size bounds
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Convergence check
                if sigma < 1e-6:
                    break

            return best_x, best_f, calls

        # Phase 1: Initial random sampling to find good starting points
        n_random = min(20, budget // 15)
        candidates = []
        for i in range(n_random):
            if time_exceeded():
                break
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            candidates.append((x, fx))

        # Sort by fitness
        candidates.sort(key=lambda t: t[1])
        best_x, best_f = candidates[0]
        calls_used = len(candidates)

        # Phase 2: Run (1+1)-ES from top candidates with restarts
        remaining = budget - calls_used
        n_restarts = min(5, max(1, remaining // 50))
        budget_per_restart = remaining // n_restarts

        for restart in range(n_restarts):
            if time_exceeded():
                break
            if calls_used >= budget:
                break

            # Choose starting point
            if restart == 0:
                x0 = best_x.copy()
                sigma0 = 2.0
            elif restart == 1 and len(candidates) > 1:
                x0 = candidates[1][0].copy()
                sigma0 = 1.5
            else:
                # Perturb from best
                x0 = best_x + rng.standard_normal(d) * 0.5
                x0 = np.clip(x0, -5, 5)
                sigma0 = 1.0

            # Run ES
            new_x, new_f, calls_made = run_es(x0, sigma0, min(budget_per_restart, budget - calls_used), rng, func)
            calls_used += calls_made

            if new_f < best_f:
                best_f = new_f
                best_x = new_x.copy()

        # Phase 3: Local refinement with smaller steps around best
        remaining = budget - calls_used
        if remaining > 10 and not time_exceeded():
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(min(remaining, 50)):
                if time_exceeded():
                    break
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                sigma = np.clip(sigma, 1e-6, 2.0)
                if sigma < 1e-5:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
