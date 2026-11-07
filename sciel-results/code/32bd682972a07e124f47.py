"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with diverse initial points."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0
        n = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)
        best_x = None
        best_fx = np.inf
        eval_count = 0

        def wrapped(x):
            nonlocal eval_count
            if eval_count >= budget:
                return 1e10
            eval_count += 1
            return func(np.clip(x, -5, 5))

        # Generate diverse starting points
        starts = []
        # Origin
        starts.append(np.zeros(n))
        # Random points
        for _ in range(10):
            starts.append(rng.uniform(-4, 4, n))
        # Axis-aligned points
        for _ in range(5):
            pt = np.zeros(n)
            pt[rng.integers(0, n)] = rng.uniform(-4, 4)
            starts.append(pt)

        # Sort by heuristic: prefer points closer to center
        starts.sort(key=lambda x: np.linalg.norm(x))

        # Run L-BFGS-B from each starting point until budget or time runs out
        for i, x0 in enumerate(starts):
            if time.time() > deadline - 2.0:
                break
            if eval_count >= budget:
                break

            # Estimate remaining budget for this run
            remaining = budget - eval_count
            maxiter = max(1, min(remaining, 50))  # Cap per-run iterations

            try:
                res = minimize(
                    wrapped, x0, method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': maxiter, 'maxls': 20}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # If we have budget left, do a final polish from the best point
        if time.time() < deadline - 1.0 and eval_count < budget and best_x is not None:
            remaining = budget - eval_count
            if remaining >= 5:
                try:
                    res = minimize(
                        wrapped, best_x, method='L-BFGS-B',
                        bounds=[(-5, 5)] * n,
                        options={'maxiter': remaining, 'maxls': 20}
                    )
                    if res.fun < best_fx:
                        best_fx = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(n)
            best_fx = func(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
