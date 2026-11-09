import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Optimizer for Bent Cigar (f12) combining Sobol global search with
    adaptive (1+1)-ES for anisotropic refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        calls_used = 0
        best_x = None
        best_fx = float('inf')

        def guarded(x):
            nonlocal calls_used, best_x, best_fx
            if time.time() >= deadline or calls_used >= budget:
                raise Exception("Limit exceeded")
            x = np.asarray(x, dtype=np.float64)
            calls_used += 1
            fx = float(func(x))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol global scan to find a good starting point
        n_scan = min(50, budget // 5)
        if n_scan < 10:
            n_scan = 10

        try:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            u = sampler.random(n_scan)
        except Exception:
            rng_scan = np.random.default_rng(42)
            u = rng_scan.random((n_scan, dim))

        pts = -5.0 + 10.0 * u

        for i in range(n_scan):
            if time.time() >= deadline or calls_used >= budget:
                break
            try:
                guarded(pts[i])
            except Exception:
                break

        if best_x is None:
            # Fallback: evaluate origin
            try:
                guarded(np.zeros(dim))
            except Exception:
                return np.zeros(dim), float('inf')

        x = best_x.copy()
        fx = best_fx

        # Phase 2: (1+1)-ES with 1/5 success rule for adaptive step size
        rng = np.random.default_rng(12345)
        sigma = 1.0
        successes = 0
        total = 0
        success_threshold = 0.2  # 1/5 rule

        while calls_used < budget:
            if time.time() >= deadline:
                break

            direction = rng.standard_normal(dim)
            y = np.clip(x + sigma * direction, -5, 5)

            try:
                fy = guarded(y)
            except Exception:
                break

            if fy < fx:
                x = y
                fx = fy
                successes += 1
            total += 1

            # Adapt step size based on success rate
            if total >= 5:
                rate = successes / total
                if rate > success_threshold:
                    sigma *= 1.3
                elif rate < success_threshold:
                    sigma *= 0.7
                successes = 0
                total = 0

            # Bounds on sigma
            if sigma < 1e-6:
                sigma = 1e-6
            if sigma > 10:
                sigma = 10

        return best_x, best_fx
