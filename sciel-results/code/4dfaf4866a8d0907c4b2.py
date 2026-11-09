"""Rewritten optimizer for BBOB f20 (Schwefel) with multi-start + L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start global search with L-BFGS-B local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        def clipped_func(x):
            return func(np.clip(x, -5, 5))

        rng = np.random.default_rng(42)

        # Phase 1: Multi-point initialization for global coverage
        # Use a combination of random points and structured points
        n_init = min(50, self.budget // 10)
        candidates = []

        # Random points
        for _ in range(n_init - 5):
            x = rng.uniform(-5, 5, self.dim)
            candidates.append((x, clipped_func(x)))

        # Structured points: corners, center, and a few strategic locations
        structured = []
        # Center
        structured.append(np.zeros(self.dim))
        # Corners (sample a few)
        for i in range(min(4, n_init - 6)):
            corner = rng.choice([-5, 5], self.dim)
            structured.append(corner)
        # A few points near typical Schwefel minima (around -4.2 each dim for standard)
        for i in range(min(5, n_init - 10)):
            x = rng.normal(-4.2, 1.5, self.dim)
            x = np.clip(x, -5, 5)
            structured.append(x)

        for x in structured:
            x = np.clip(x, -5, 5)
            candidates.append((x, clipped_func(x)))

        # Sort by function value
        candidates.sort(key=lambda t: t[1])

        best_x = candidates[0][0].copy()
        best_f = candidates[0][1]

        # Phase 2: Local refinement using L-BFGS-B from top candidates
        n_local = min(10, len(candidates), self.budget - len(candidates))
        used_budget = len(candidates)

        for i in range(n_local):
            if time.time() > deadline:
                break

            x0 = candidates[i][0].copy()
            try:
                res = minimize(
                    clipped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                used_budget += res.nfev
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()

                # Update candidate list with the result
                candidates[i] = (res.x.copy(), res.fun)
            except Exception:
                continue

        # Phase 3: If budget remains, do a (1+1)-ES refinement from best point
        remaining = self.budget - used_budget
        if remaining > 20 and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(min(remaining, 200)):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = clipped_func(y)
                if fy < fx:
                    x, fx = y, fy
                    best_x, best_f = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9
                    if sigma < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
