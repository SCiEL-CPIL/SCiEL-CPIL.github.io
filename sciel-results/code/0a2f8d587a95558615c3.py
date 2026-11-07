"""L-BFGS-B multi-restart optimizer for Bent Cigar (f12)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart L-BFGS-B for Bent Cigar: exploits curvature in anisotropic valleys."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        calls = [0]
        best_x = None
        best_f = np.inf
        deadline = time.time() + 280  # safety margin before 330s kill

        def F(x):
            calls[0] += 1
            if calls[0] > budget:
                return 1e10
            val = func(np.asarray(x, dtype=float))
            return val

        bounds = [(-5.0, 5.0)] * d
        rng = np.random.default_rng(42)

        # Number of restarts: use remaining budget wisely
        # Each L-BFGS-B run on 5-dim typically uses ~30-80 evals
        # With 500 budget, we can afford ~5-10 restarts
        n_restarts = 6

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if calls[0] >= budget - 10:
                break

            # Generate diverse starting points
            if r == 0:
                x0 = np.zeros(d)
            elif r == 1:
                x0 = rng.uniform(-2.0, 2.0, d)
            elif r == 2:
                x0 = rng.uniform(-4.0, 4.0, d)
            else:
                x0 = rng.uniform(-5.0, 5.0, d)

            # Clip to bounds
            x0 = np.clip(x0, -5.0, 5.0)

            # Evaluate starting point
            f0 = F(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            if calls[0] >= budget - 10:
                break

            # L-BFGS-B with bounded variables
            # maxiter controls iterations, but we track actual function calls
            try:
                res = minimize(
                    F, x0, method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 200,
                        'maxfun': max(1, budget - calls[0] - 5),
                        'gtol': 1e-12,
                        'ftol': 1e-14,
                    }
                )
            except Exception:
                break

            if res.fun < best_f:
                best_f = res.fun
                best_x = np.clip(res.x, -5.0, 5.0)

        if best_x is None:
            best_x = np.zeros(d)
            best_f = F(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
