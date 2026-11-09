"""Surrogate-assisted optimizer for BBOB f12 (Bent Cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator
import time


class Optimizer:
    """Multi-fidelity surrogate-assisted search using RBF + L-BFGS-B."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Storage for observed points
        X = []
        Y = []

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def evaluate(x):
            x = clip(x)
            fx = func(x)
            X.append(x.copy())
            Y.append(fx)
            return fx

        # Phase 1: Initial sampling to build a rough surrogate
        # Use a mixture of random and structured points
        rng = np.random.default_rng(42)

        # Start with a few random points
        n_initial = min(20, budget)
        for _ in range(n_initial):
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        # Add some structured points (axis-aligned) to help capture anisotropy
        for axis in range(dim):
            for val in [-4.0, -2.0, 0.0, 2.0, 4.0]:
                if len(X) >= budget:
                    break
                x = np.zeros(dim)
                x[axis] = val
                evaluate(x)

        best_x = X[np.argmin(Y)]
        best_f = min(Y)

        # Phase 2: Iterative surrogate-assisted optimization
        call_count = len(X)
        min_points_for_rbf = dim + 2

        while call_count < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Need enough points for RBF
            if len(X) < min_points_for_rbf:
                x = rng.uniform(-5, 5, dim)
                evaluate(x)
                call_count += 1
                continue

            # Build RBF surrogate
            X_arr = np.array(X)
            Y_arr = np.array(Y)

            try:
                rbf = RBFInterpolator(X_arr, Y_arr, kernel='thin_plate_spline')
            except Exception:
                # Fallback: just do random sampling
                x = rng.uniform(-5, 5, dim)
                evaluate(x)
                call_count += 1
                continue

            # Define surrogate function for L-BFGS-B
            def surrogate(x):
                x = clip(x)
                try:
                    return float(rbf(x.reshape(1, -1))[0])
                except Exception:
                    return 1e10

            # Try to find minimum of surrogate using L-BFGS-B
            # Use multiple starting points to avoid local minima
            n_starts = 3
            surrogate_best_x = None
            surrogate_best_f = 1e18

            for i in range(n_starts):
                if i == 0:
                    x0 = best_x.copy()
                elif i == 1:
                    x0 = rng.uniform(-5, 5, dim)
                else:
                    # Perturb best
                    x0 = clip(best_x + 0.5 * rng.standard_normal(dim))

                try:
                    res = minimize(
                        surrogate,
                        x0,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if res.success or res.fun < surrogate_best_f:
                        surrogate_best_f = res.fun
                        surrogate_best_x = clip(res.x)
                except Exception:
                    pass

            if surrogate_best_x is not None:
                # Evaluate the surrogate-predicted best point
                x_new = surrogate_best_x
                f_new = evaluate(x_new)
                call_count += 1

                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
            else:
                # Fallback: random step
                x = rng.uniform(-5, 5, dim)
                evaluate(x)
                call_count += 1

            # Occasionally add a random point to explore
            if call_count < budget and rng.random() < 0.1:
                x = rng.uniform(-5, 5, dim)
                evaluate(x)
                call_count += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
