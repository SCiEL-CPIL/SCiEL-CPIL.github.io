"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Global exploration with L-BFGS-B local refinement for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        from scipy.optimize import minimize
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270.0
        rng = np.random.default_rng(12345)

        best_x = np.zeros(dim)
        best_f = float('inf')
        call_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget:
                return float('inf')
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        def time_ok():
            return time.time() < deadline

        # Phase 1: Global exploration — random sampling to get into the discus valley
        # Discus f11 has a flat valley; we need to find the narrow basin
        global_budget = int(budget * 0.4)

        # Start with origin and a few structured points
        candidates = [np.zeros(dim)]
        # Add random points in the box
        for _ in range(20):
            candidates.append(rng.uniform(-4, 4, dim))
        # Add axis-aligned points
        for i in range(dim):
            for val in [-2.0, 0.0, 2.0]:
                p = np.zeros(dim)
                p[i] = val
                candidates.append(p)

        for c in candidates:
            if call_count >= global_budget or not time_ok():
                break
            evaluate(c)

        # Random restarts in the remaining global budget
        while call_count < global_budget and time_ok():
            x = rng.uniform(-4.5, 4.5, dim)
            evaluate(x)

        # Phase 2: L-BFGS-B local refinement from the best point and a few other good points
        # Gather the top few distinct starting points
        # We'll just use the best point and do a few local optimizations
        local_budget = budget - call_count

        if local_budget > 5 and time_ok():
            # Try L-BFGS-B from the current best
            def wrapper(x):
                if not time_ok() or call_count >= budget:
                    return float('inf')
                return evaluate(x)

            try:
                res = minimize(wrapper, best_x, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0)] * dim,
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8,
                                        'maxls': 30})
            except Exception:
                pass

            # If we have budget left, try from a few other points
            if call_count < budget and time_ok():
                for attempt in range(3):
                    if call_count >= budget - 10 or not time_ok():
                        break
                    # Perturb the best point slightly
                    x0 = best_x + rng.normal(0, 0.1, dim)
                    x0 = np.clip(x0, -5.0, 5.0)
                    try:
                        res2 = minimize(wrapper, x0, method='L-BFGS-B',
                                        bounds=[(-5.0, 5.0)] * dim,
                                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8,
                                                 'maxls': 30})
                    except Exception:
                        pass

        return best_x, best_f

# EVOLVE-BLOCK-END
