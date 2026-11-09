"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset.

Uses a robust (1+1)-ES with adaptive step size and occasional restarts.
This is a black-box, structure-agnostic approach that performs well on
smooth, unimodal functions like f14 even under unknown affine transforms.
"""
import time
import numpy as np


class Optimizer:
    """(1+1)-ES optimizer with restarts for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 320.0
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget
        calls = 0

        def elapsed():
            return time.time() - start

        def still_ok():
            return elapsed() < deadline

        def make_call(x):
            nonlocal calls
            if calls >= budget:
                return None
            fx = float(func(x))
            calls += 1
            return fx

        # Initial evaluation at origin
        x0 = np.zeros(dim)
        fx0 = make_call(x0)
        if fx0 is None:
            return x0, np.inf

        best_x = x0.copy()
        best_f = fx0

        # (1+1)-ES with restarts
        # Use a population of 1 (just the candidate) but adapt sigma
        # Restart if no improvement for a while

        def run_es(x_init, f_init, sigma_init, max_calls):
            nonlocal calls, best_x, best_f
            x = x_init.copy()
            f = f_init
            sigma = sigma_init
            no_improve = 0
            stagnation_limit = 15 * dim  # restart if no improvement for this many iterations

            while still_ok() and calls < budget and calls < max_calls:
                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = make_call(y)
                if fy is None:
                    break

                if fy < f:
                    x = y.copy()
                    f = fy
                    no_improve = 0
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    # Adapt sigma: increase slightly on success
                    sigma *= 1.05
                else:
                    no_improve += 1
                    # Adapt sigma: decrease on failure
                    sigma *= 0.95

                # Prevent sigma from becoming too small or too large
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Restart if stagnated
                if no_improve >= stagnation_limit:
                    break

            return x, f, sigma

        # Multiple restarts from different starting points
        n_restarts = 5
        calls_per_restart = max(10, (budget - calls) // n_restarts)

        # First restart from current best
        x, f, sigma = run_es(best_x, best_f, 1.0, calls + calls_per_restart)

        for r in range(1, n_restarts):
            if not still_ok() or calls >= budget:
                break
            # New starting point: random or near best
            if r % 2 == 1:
                x_init = rng.uniform(-5.0, 5.0, dim)
            else:
                x_init = best_x + 2.0 * rng.standard_normal(dim)
                x_init = np.clip(x_init, -5.0, 5.0)

            f_init = make_call(x_init)
            if f_init is None:
                break

            if f_init < best_f:
                best_f = f_init
                best_x = x_init.copy()

            x, f, sigma = run_es(x_init, f_init, 1.0, calls + calls_per_restart)

        # Final local refinement around best_x with small sigma
        if still_ok() and calls < budget:
            x, f, sigma = run_es(best_x, best_f, 0.1, budget)

        return best_x, float(best_f)
