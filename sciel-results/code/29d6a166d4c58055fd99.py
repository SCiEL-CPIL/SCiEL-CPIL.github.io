"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with covariance adaptation, 1/5 rule, and L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0
        n = self.dim
        budget = self.budget
        # Reserve last ~30 evals for local polish
        polish_budget = min(30, budget // 10)
        es_budget = budget - polish_budget

        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, n)
        fx = func(x)
        best_x, best_fx = x.copy(), fx

        sigma = 2.0
        C = np.eye(n)
        # 1/5 success rule with exponential moving average
        successes = 0
        total = 0
        c_sigma = 0.5
        c_c = 1.0 / (n + 3)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        for i in range(es_budget):
            if time.time() > deadline - 5.0:
                break

            # Generate candidate using current covariance
            u = rng.multivariate_normal(np.zeros(n), C, size=1)
            y = np.clip(x + sigma * u, -5, 5)
            fy = func(y)

            # Update covariance with rank-1 and rank-2 (simplified CMA)
            if fy < fx:
                x, fx = y, fy
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
                successes += 1
                # Rank-1 update
                C = (1 - c_c) * C + c_c * np.outer(u, u)
                # Normalize to keep trace
                C /= np.trace(C) / n
            else:
                # Rank-2 update on failure (helps adapt away from bad directions)
                C = (1 - c_c * 0.5) * C + c_c * 0.5 * np.outer(u, u)
                C /= np.trace(C) / n

            total += 1
            # 1/5 success rule for sigma
            if total >= 10:
                rate = successes / total
                if rate > 0.2:
                    sigma *= 1.05
                elif rate < 0.2:
                    sigma *= 0.95
                successes = 0
                total = 0

            # Clamp sigma
            sigma = np.clip(sigma, 1e-6, 10.0)

        # Final polish with L-BFGS-B
        if time.time() < deadline - 1.0 and polish_budget > 0:
            def wrapped(x):
                return func(np.clip(x, -5, 5))

            try:
                res = minimize(
                    wrapped, best_x, method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': polish_budget, 'maxls': 20}
                )
                if res.fun < best_fx:
                    best_x = np.clip(res.x, -5, 5)
                    best_fx = res.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
