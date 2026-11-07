"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _sobol_5d(n):
    """Generate n quasi-random points in [0,1)^5 using a simple Halton sequence."""
    def halton(index, base):
        f, r = 1.0, 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    points = np.empty((n, 5))
    for j in range(5):
        base = j + 2  # bases 2,3,4,5,6
        for i in range(n):
            points[i, j] = halton(i + 1, base)
    return points


class Optimizer:
    """Sobol global scan + (1+1)-ES refinement with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # 300s limit, 20s safety margin
        dim = self.dim
        budget = self.budget

        # Phase 1: Sobol global scan (up to 50 calls)
        n_sobol = min(50, budget // 10)
        points = _sobol_5d(n_sobol)
        # Map to [-5, 5]^dim
        points = points * 10.0 - 5.0

        best_x = None
        best_f = np.inf

        for i in range(n_sobol):
            if time.time() > deadline:
                break
            x = points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES refinement from best point
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        x = best_x.copy()
        fx = best_f
        sigma = 1.0

        rng = np.random.default_rng(42)
        calls_made = n_sobol

        while calls_made < budget:
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.25
            else:
                sigma *= 0.85
                if sigma < 1e-10:
                    break

        return x, fx
# EVOLVE-BLOCK-END
