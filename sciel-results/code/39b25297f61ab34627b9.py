"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """Population-based CMA-ES with restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Use a deterministic seed for reproducibility
        rng = np.random.default_rng(42)

        # Track best solution found
        best_x = None
        best_f = np.inf
        calls_made = 0

        def count_call():
            nonlocal calls_made
            calls_made += 1

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            count_call()
            if calls_made > budget:
                raise ValueError("Budget exceeded")
            val = func(x)
            if best_x is None or val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Initial random sampling to find good starting points
        n_initial = 20
        initial_points = []
        for i in range(n_initial):
            if calls_made >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = wrapped_func(x)
            initial_points.append((x, f))

        if not initial_points:
            return best_x, best_f

        # Sort initial points by fitness
        initial_points.sort(key=lambda t: t[1])

        # Phase 2: Run multiple local optimizations from the best starting points
        # Use Nelder-Mead and other local methods

        local_starts = initial_points[:5]  # Top 5 starting points

        for start_x, start_f in local_starts:
            if calls_made >= budget or time.time() > deadline:
                break

            # Try Nelder-Mead
            try:
                result = minimize(
                    wrapped_func,
                    start_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

            if calls_made >= budget or time.time() > deadline:
                break

            # Try Powell method
            try:
                result = minimize(
                    wrapped_func,
                    start_x,
                    method='Powell',
                    options={'maxiter': 50, 'xtol': 1e-6, 'ftol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

            if calls_made >= budget or time.time() > deadline:
                break

        # Phase 3: If we still have budget, do a fine-grained local search around the best point
        if calls_made < budget and time.time() < deadline:
            # Use a (1+1)-ES with adaptive step size for final polishing
            x = best_x.copy()
            f = best_f
            sigma = 0.1  # Start with small step size for polishing

            # Adaptive (1+1)-ES
            no_improve = 0
            max_no_improve = 30

            while calls_made < budget and time.time() < deadline:
                if no_improve > max_no_improve:
                    break

                # Generate mutant
                mutant = x + sigma * rng.standard_normal(dim)
                mutant = np.clip(mutant, -5, 5)

                try:
                    f_mutant = wrapped_func(mutant)
                except Exception:
                    break

                if f_mutant < f:
                    x = mutant
                    f = f_mutant
                    best_f = f
                    best_x = x.copy()
                    no_improve = 0
                    # Increase sigma slightly on success
                    sigma *= 1.1
                else:
                    no_improve += 1
                    # Decrease sigma on failure
                    sigma *= 0.95

                # Clamp sigma
                sigma = np.clip(sigma, 1e-6, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
