"""Two-phase hybrid: dense deterministic axis/radial sweep to identify the Discus needle,
then L-BFGS-B with multi-start for rapid exploitation of the anisotropic landscape."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Two-phase optimizer: deterministic sweep + L-BFGS-B for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 330s limit
        rng = np.random.default_rng(42)

        # Track best
        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls
            if calls >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Dense deterministic sweep to find the narrow axis of Discus
        # BBOB f11 (Discus): f(x) = 0.5 * (||x||^2 + 2 * max_i |x_i|^2)
        # The needle is along the axis where the max component is large.

        # First, evaluate origin
        evaluate(np.zeros(dim))

        # Axis-based sweep: for each axis, try positive and negative directions at several radii
        radii = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        for axis in range(dim):
            for r in radii:
                if calls >= budget or time.time() > deadline:
                    break
                # Positive direction
                x = np.zeros(dim)
                x[axis] = r
                evaluate(x)
                # Negative direction
                x[axis] = -r
                if calls < budget and time.time() < deadline:
                    evaluate(x)
            if calls >= budget or time.time() > deadline:
                break

        # Radial sweep in random directions
        n_radial = 20
        for i in range(n_radial):
            if calls >= budget or time.time() > deadline:
                break
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction)
            for r in [1.0, 2.0, 3.0, 4.0, 5.0]:
                if calls >= budget or time.time() > deadline:
                    break
                evaluate(direction * r)

        # Phase 2: L-BFGS-B with multi-start for rapid convergence
        if calls < budget and best_x is not None:
            self._lbfgsb_multistart(func, best_x, budget - calls, deadline, rng, evaluate)

        return best_x, best_f

    def _lbfgsb_multistart(self, func, x0, max_evals, deadline, rng, evaluate):
        """Run L-BFGS-B with multiple restarts starting from diverse points."""
        dim = self.dim

        # Generate multiple starting points around the best point found
        start_points = [x0.copy()]

        # Add perturbed versions of the best point
        for _ in range(5):
            perturbation = rng.standard_normal(dim) * 0.5
            x_pert = np.clip(x0 + perturbation, -5.0, 5.0)
            start_points.append(x_pert)

        # Add points along each axis from the best point
        for axis in range(dim):
            for sign in [-1, 1]:
                x_pert = x0.copy()
                x_pert[axis] += sign * 1.0
                x_pert = np.clip(x_pert, -5.0, 5.0)
                start_points.append(x_pert)

        # Shuffle start points to diversify
        rng.shuffle(start_points)

        for x_start in start_points:
            if calls := self._get_calls() >= max_evals:
                break
            if time.time() > deadline:
                break

            # Run L-BFGS-B from this starting point
            self._lbfgsb_single(func, x_start, deadline, evaluate)

    def _get_calls(self):
        return self._calls

    def _lbfgsb_single(self, func, x_start, deadline, evaluate):
        """Run a single L-BFGS-B optimization from x_start."""
        # Wrap the function to track calls and enforce deadline
        call_count = [0]

        def wrapped_func(x):
            nonlocal call_count
            if call_count[0] >= 50:  # Limit evaluations per L-BFGS-B run
                return np.inf
            if time.time() > deadline:
                return np.inf
            fx = evaluate(x)
            call_count[0] += 1
            return fx

        try:
            result = minimize(
                wrapped_func,
                x_start,
                method='L-BFGS-B',
                bounds=[(-5.0, 5.0)] * self.dim,
                options={'maxiter': 100, 'maxfun': 50}
            )
            if result.fun < np.inf:
                evaluate(result.x)
        except Exception:
            pass
# EVOLVE-BLOCK-END
