"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B optimizer for BBOB-style f19."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Track function calls to enforce budget
        call_count = 0

        def counted_func(x):
            nonlocal call_count
            call_count += 1
            return func(x)

        # Generate diverse starting points
        # 1. Origin
        # 2. Stratified random points (LHS-like)
        # 3. Points near boundaries
        n_starts = 10
        starts = []
        starts.append(np.zeros(n))
        
        # Add stratified points
        for i in range(n_starts - 1):
            # Stratified sampling in each dimension
            x = np.array([(i + rng.random()) / n_starts * 10 - 5 for _ in range(n)])
            starts.append(x)
            
        # Add a few random points for robustness
        for _ in range(5):
            starts.append(rng.uniform(-5, 5, size=n))

        best_x = None
        best_f = np.inf
        bounds = [(-5, 5)] * n

        for x0 in starts:
            if time.time() > deadline:
                break
            if call_count >= budget:
                break

            # Limit evaluations per run to allow multiple restarts
            # L-BFGS-B is efficient; 40 evals per run is reasonable
            remaining = budget - call_count
            max_evals_per_run = min(remaining, 40)
            if max_evals_per_run < 10:
                break

            try:
                res = minimize(counted_func, x0, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 100, 'maxfun': max_evals_per_run})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # If we still have budget, do a final polish on the best point
        if best_x is not None and call_count < budget and time.time() < deadline:
            remaining = budget - call_count
            if remaining > 10:
                try:
                    res = minimize(counted_func, best_x, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': 100, 'maxfun': remaining})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
