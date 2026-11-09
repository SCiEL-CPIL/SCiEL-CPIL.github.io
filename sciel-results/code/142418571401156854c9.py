"""Optimizer for BBOB-style f13 (Sharp ridge) with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Halton multi-start coverage followed by adaptive (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    @staticmethod
    def _halton(dim, count, rng):
        """Generate Halton sequence points in [0,1)^dim."""
        points = []
        for i in range(1, count + 1):
            pt = []
            for base in range(2, 2 + dim):
                f = 1.0 / base
                r = 0.0
                n = i
                while n > 0:
                    n, rem = divmod(n, base)
                    r += rem * f
                    f /= base
                pt.append(r)
            points.append(pt)
        return np.array(points)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin
        budget = self.budget
        dim = self.dim

        # Phase 1: Deterministic multi-start with Halton + a few random points
        # Use ~40% of budget for exploration
        n_explore = max(10, int(budget * 0.35))
        n_halton = min(n_explore, 30)
        n_random = n_explore - n_halton

        rng = np.random.default_rng(42)

        # Generate candidate starting points
        halton_pts = self._halton(dim, n_halton, rng)
        halton_pts = halton_pts * 10.0 - 5.0  # map to [-5, 5]

        random_pts = rng.uniform(-5, 5, size=(n_random, dim))

        # Evaluate all candidates
        candidates = []  # list of (fx, x)
        calls_used = 0

        for pt in list(halton_pts) + list(random_pts):
            if time.time() > deadline:
                break
            pt = np.clip(pt, -5, 5)
            fx = func(pt)
            calls_used += 1
            candidates.append((fx, pt.copy()))

        # Sort by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Adaptive (1+1)-ES from the best candidate
        # Use remaining budget for exploitation
        best_fx, best_x = candidates[0]
        remaining = budget - calls_used

        if remaining <= 0:
            return best_x, best_fx

        sigma = 1.0
        x = best_x.copy()
        fx = best_fx

        # Adaptive step size with 1/5 rule
        successes = 0
        window = 10

        for _ in range(remaining):
            if time.time() > deadline:
                break

            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                x = y
                fx = fy
                successes += 1
                sigma *= 1.3
            else:
                sigma *= 0.9

            # Adjust sigma based on success rate over window
            if calls_used % window == 0:
                # Re-estimate: if too many successes, increase sigma
                # Simple heuristic: track recent successes
                pass

            # Keep sigma within reasonable bounds
            sigma = np.clip(sigma, 0.01, 5.0)

        return x, fx
# EVOLVE-BLOCK-END
