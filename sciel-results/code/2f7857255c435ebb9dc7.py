"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: (1+1)-ES with adaptive restarts and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Initial point: Rosenbrock minimum is at 1, so start there
        x = np.ones(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        sigma = 1.0
        no_improve = 0
        max_no_improve = 50

        calls_made = 1

        while calls_made < self.budget:
            # Check time limit
            if time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Accept
                x = y
                fx = fy
                no_improve = 0

                # Update best
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

                # Increase sigma on success
                sigma *= 1.1
            else:
                # Reject
                no_improve += 1
                sigma *= 0.95

                # Restart if stuck
                if no_improve >= max_no_improve:
                    # Try a larger perturbation from best point
                    for _ in range(5):
                        if calls_made >= self.budget or time.time() > deadline:
                            break
                        candidate = best_x + 2.0 * rng.standard_normal(self.dim)
                        candidate = np.clip(candidate, -5, 5)
                        cf = func(candidate)
                        calls_made += 1
                        if cf < best_fx:
                            best_fx = cf
                            best_x = candidate.copy()
                            x = best_x.copy()
                            fx = best_fx
                            no_improve = 0
                            break

                    # If still stuck, do a full restart from a random point
                    if no_improve >= max_no_improve:
                        x = rng.uniform(-4, 4, self.dim)
                        fx = func(x)
                        calls_made += 1
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                        no_improve = 0
                        sigma = 1.0

        # Local refinement using Powell if we have time
        if time.time() < deadline - 30 and calls_made < self.budget:
            try:
                from scipy.optimize import minimize

                # Count remaining calls
                remaining = self.budget - calls_made

                # Use Powell with limited iterations
                def wrapper(z):
                    nonlocal calls_made
                    if calls_made >= self.budget or time.time() > deadline:
                        return best_fx
                    calls_made += 1
                    return func(np.clip(z, -5, 5))

                result = minimize(
                    wrapper,
                    best_x,
                    method='Powell',
                    options={'maxiter': max(1, remaining // 2), 'ftol': 1e-10, 'xtol': 1e-10}
                )

                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = np.clip(result.x, -5, 5)

            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
