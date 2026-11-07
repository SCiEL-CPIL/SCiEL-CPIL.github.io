"""Rewritten optimizer: Sobol global exploration + multi-scale Gaussian exploitation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Sobol global search followed by multi-scale Gaussian hill-climbing."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _sobol_points(self, n, seed=0):
        """Deterministic low-discrepancy Sobol points mapped to [-5,5]^dim."""
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=seed)
        # next() requires n to be a power of two; round up.
        n_pow2 = 1
        while n_pow2 < n:
            n_pow2 *= 2
        pts = sampler.random(n_pow2)[:n]
        return 10.0 * pts - 5.0  # map [0,1] -> [-5,5]

    def __call__(self, func):
        start = time.time()
        deadline = start + 290.0  # safety margin vs 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf

        def consider(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x
            return f < best_f

        # ---- Phase 1: deterministic Sobol global exploration ----
        # Use a fixed number of well-spread points; the rest of the budget
        # goes to exploitation. Keep it modest so wall-clock stays low.
        n_explore = min(64, budget)
        for x in self._sobol_points(n_explore):
            if time.time() > deadline:
                break
            f = func(x)
            consider(x, f)

        # ---- Phase 2: multi-scale Gaussian exploitation around the best ----
        if best_x is None:
            # fallback: uniform random
            for _ in range(min(budget - n_explore, 32)):
                if time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                consider(x, func(x))
            return best_x, best_f

        x = best_x.copy()
        fx = best_f
        remaining = budget - n_explore
        if remaining <= 0:
            return best_x, best_f

        # Multi-scale: start with a moderately large step, decay it so we
        # refine from coarse to fine. Weierstrass has structure at all scales,
        # so decaying step size lets us descend through nested minima.
        sigma = 1.5
        for step in range(remaining):
            if time.time() > deadline:
                break
            # Occasionally inject a larger perturbation to escape local minima.
            if step % 20 == 0 and step > 0:
                sigma = max(sigma, 0.8)
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3          # reward: expand a little
                consider(x, fx)
            else:
                sigma *= 0.85         # shrink
                sigma = max(sigma, 1e-4)

        return best_x, best_f
# EVOLVE-BLOCK-END
