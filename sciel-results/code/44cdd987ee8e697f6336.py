"""Rewritten optimizer using Brent's method coordinate descent with adaptive budgeting."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar


class Optimizer:
    """Coordinate descent using Brent's method for 1D line minimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Initialize with a few random starts to get a good starting point
        # Use a small number of evaluations to find a decent initial point
        n_init = min(10, budget // 10)
        best_x = None
        best_f = np.inf

        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        calls_used = n_init
        remaining = budget - calls_used

        # Coordinate descent with Brent's method
        # Each iteration cycles through all dimensions
        # We need to be careful about budget usage in Brent's method

        # Estimate how many evaluations Brent's method will use per coordinate
        # Typically 10-20 evaluations for good convergence
        # We'll do multiple passes, reducing the number of coordinates per pass if budget is low

        # Strategy: do full passes until budget is exhausted
        # Each coordinate minimization uses ~15 evaluations on average
        evals_per_coord = 15
        coords_per_pass = max(1, min(dim, remaining // evals_per_coord))

        while remaining > 0:
            # Determine which coordinates to try in this pass
            # Try all coordinates if we have enough budget, otherwise a subset
            if remaining >= dim * evals_per_coord:
                coords_to_try = list(range(dim))
            else:
                # Cycle through coordinates, prioritizing those that might help most
                # Simple approach: try all but with fewer evaluations
                coords_to_try = list(range(dim))

            for d in coords_to_try:
                if remaining < 5:
                    break

                # Define the 1D function for this coordinate
                x_base = best_x.copy()

                def f1d(t):
                    nonlocal best_x
                    x_test = x_base.copy()
                    x_test[d] = t
                    return func(x_test)

                # Use current value as center, search in [-5, 5]
                # But we can narrow the search range based on current best
                # For safety, use the full box but let Brent's method converge
                center = x_base[d]

                # Search interval: clip to [-5, 5]
                # Use a reasonable interval around the current value
                # If we're near the boundary, adjust
                if center > 3:
                    bracket = (center - 4, 5)
                elif center < -3:
                    bracket = (-5, center + 4)
                else:
                    bracket = (center - 2, center + 2)
                    # Ensure bracket is within [-5, 5]
                    bracket = (max(-5, bracket[0]), min(5, bracket[1]))

                # Make sure the bracket is valid
                if bracket[0] >= bracket[1]:
                    bracket = (-5, 5)

                try:
                    # Use minimize_scalar with Brent's method
                    # Limit the number of function evaluations
                    maxiter = min(50, remaining)
                    result = minimize_scalar(
                        f1d,
                        bounds=bracket,
                        method='bounded',
                        options={'maxiter': maxiter}
                    )

                    # Update best if we found something better
                    if result.fun < best_f:
                        best_x[d] = result.x
                        best_f = result.fun

                    # Count approximate evaluations used
                    # minimize_scalar with bounded method uses ~maxiter evaluations
                    # But it may use fewer if it converges early
                    # We'll estimate conservatively
                    evals_used = min(maxiter, 20)
                    remaining -= evals_used

                except Exception:
                    # If Brent's method fails, try a simple coordinate step
                    step = 0.5
                    x_plus = x_base.copy()
                    x_plus[d] = min(5, x_base[d] + step)
                    x_minus = x_base.copy()
                    x_minus[d] = max(-5, x_base[d] - step)

                    f_plus = func(x_plus)
                    f_minus = func(x_minus)
                    remaining -= 2

                    if f_plus < best_f:
                        best_x[d] = x_plus[d]
                        best_f = f_plus
                    elif f_minus < best_f:
                        best_x[d] = x_minus[d]
                        best_f = f_minus

            # After a full pass, check if we should continue
            if remaining < dim * 5:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
