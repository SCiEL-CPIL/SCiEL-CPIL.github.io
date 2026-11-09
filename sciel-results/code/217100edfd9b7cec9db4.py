"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with finite-difference gradients on rotated Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin

        # Count calls carefully
        call_count = [0]
        max_calls = budget

        def counted_func(x):
            if call_count[0] >= max_calls:
                return 1e10
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            call_count[0] += 1
            return val

        # Finite-difference gradient with small step
        eps = 1e-5

        def grad(x):
            g = np.zeros(dim)
            for i in range(dim):
                xp = x.copy()
                xm = x.copy()
                xp[i] += eps
                xm[i] -= eps
                # Clip to box for evaluation
                xp_clipped = np.clip(xp, -5.0, 5.0)
                xm_clipped = np.clip(xm, -5.0, 5.0)
                g[i] = (counted_func(xp_clipped) - counted_func(xm_clipped)) / (2 * eps)
            return g

        # Multi-start with diverse initial points
        rng = np.random.default_rng(42)

        # Generate a set of starting points:
        # 1. Center
        # 2. Grid-like points
        # 3. Random points
        starts = []
        starts.append(np.zeros(dim))
        starts.append(np.ones(dim) * 2.0)
        starts.append(np.ones(dim) * -2.0)
        starts.append(np.ones(dim) * 3.0)
        starts.append(np.ones(dim) * -3.0)
        # Add random points
        for _ in range(20):
            starts.append(rng.uniform(-4, 4, dim))

        # Shuffle starts with a fixed seed for reproducibility
        indices = np.arange(len(starts))
        rng.shuffle(indices)
        starts = [starts[i] for i in indices]

        best_x = None
        best_f = np.inf
        bounds = [(-5.0, 5.0)] * dim

        for start in starts:
            if call_count[0] >= max_calls:
                break
            if time.time() > deadline:
                break

            # Check if we've already been near this point
            if best_x is not None and np.linalg.norm(start - best_x) < 0.1:
                continue

            try:
                res = minimize(
                    counted_func,
                    start,
                    jac=grad,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

            if call_count[0] >= max_calls:
                break
            if time.time() > deadline:
                break

        # If we still have budget, do a few more random restarts from around the best
        if best_x is not None and call_count[0] < max_calls and time.time() < deadline:
            for _ in range(10):
                if call_count[0] >= max_calls:
                    break
                if time.time() > deadline:
                    break
                perturbation = rng.uniform(-1, 1, dim) * 0.5
                new_start = np.clip(best_x + perturbation, -5, 5)
                try:
                    res = minimize(
                        counted_func,
                        new_start,
                        jac=grad,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        # Final local refinement if budget remains
        if best_x is not None and call_count[0] < max_calls and time.time() < deadline:
            for _ in range(5):
                if call_count[0] >= max_calls:
                    break
                if time.time() > deadline:
                    break
                perturbation = rng.uniform(-0.1, 0.1, dim)
                new_start = np.clip(best_x + perturbation, -5, 5)
                try:
                    res = minimize(
                        counted_func,
                        new_start,
                        jac=grad,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 50}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
