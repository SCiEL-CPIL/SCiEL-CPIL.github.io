"""Optimizer for BBOB-style Schwefel f20 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B + (1+1)-ES polish with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget:
                return best_f
            fx = func(np.asarray(x, dtype=np.float64))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, copy=True)
            return fx

        # Phase 1: Multi-start L-BFGS-B
        # L-BFGS-B converges fast on smooth landscapes like Schwefel
        n_starts = 8
        start_points = []

        # Center
        start_points.append(np.zeros(self.dim))
        # Axis-aligned points at various magnitudes
        for i in range(self.dim):
            p = np.zeros(self.dim)
            p[i] = 3.0
            start_points.append(p)
        # Additional random points
        while len(start_points) < n_starts:
            p = rng.uniform(-4, 4, self.dim)
            if all(np.linalg.norm(p - q) > 1.5 for q in start_points):
                start_points.append(p)

        # Allocate budget: ~70% to L-BFGS-B, ~30% to ES polish
        lbfgs_budget = int(self.budget * 0.7)
        per_start_budget = max(15, lbfgs_budget // n_starts)

        bounds = [(-5.0, 5.0)] * self.dim

        for sp in start_points:
            if call_count >= self.budget:
                break
            if time.time() > deadline:
                break

            lbfgs_calls_this = [0]

            def lbfgs_obj(x):
                nonlocal lbfgs_calls_this
                if lbfgs_calls_this[0] >= per_start_budget:
                    return best_f
                lbfgs_calls_this[0] += 1
                return eval_func(x)

            try:
                res = minimize(
                    lbfgs_obj,
                    sp,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': per_start_budget,
                        'ftol': 1e-8,
                        'gtol': 1e-6,
                    }
                )
            except Exception:
                pass

        # Phase 2: (1+1)-ES polish from best point found
        if best_x is not None and call_count < self.budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3

            for _ in range(self.budget - call_count):
                if time.time() > deadline:
                    break
                if call_count >= self.budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = eval_func(y)

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1
                    if sigma > 2.0:
                        sigma = 2.0
                else:
                    sigma *= 0.92
                    if sigma < 0.02:
                        # Try a moderate jump to escape local minimum
                        jump = rng.uniform(-1.5, 1.5, self.dim)
                        y2 = np.clip(x + jump, -5, 5)
                        fy2 = eval_func(y2)
                        if fy2 < fx:
                            x = y2
                            fx = fy2
                            sigma = 0.8
                        else:
                            sigma = max(sigma * 0.85, 0.008)

        # Final: use remaining budget for small perturbations of best
        if call_count < self.budget and best_x is not None:
            remaining = self.budget - call_count
            for _ in range(min(remaining, 30)):
                if time.time() > deadline:
                    break
                if call_count >= self.budget:
                    break
                y = np.clip(best_x + 0.05 * rng.standard_normal(self.dim), -5, 5)
                fy = eval_func(y)

        return best_x, best_f
# EVOLVE-BLOCK-END
