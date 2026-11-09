"""Optimizer for BBOB-style f23 (Katsuura) using multi-start CMA-ES with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start CMA-ES (via scipy) with Latin Hypercube initialization and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls = 0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(123)

        # Track calls with a wrapper that counts and respects budget
        call_count = [0]
        best_x = [None]
        best_f = [np.inf]

        def counting_func(x):
            if call_count[0] >= budget:
                return best_f[0]
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            call_count[0] += 1
            fx = func(x_clipped)
            if fx < best_f[0]:
                best_f[0] = fx
                best_x[0] = x_clipped.copy()
            return fx

        # Latin Hypercube sampling for diverse initial points
        def lhs_sample(n_samples, dim, rng):
            """Generate n_samples points via Latin Hypercube Sampling in [-5,5]^dim."""
            points = np.zeros((n_samples, dim))
            for d in range(dim):
                perm = rng.permutation(n_samples)
                points[:, d] = (perm + rng.uniform(size=n_samples)) / n_samples * 10.0 - 5.0
            return points

        # Phase 1: Latin Hypercube sampling (use ~20% of budget)
        n_lhs = max(10, min(50, budget // 10))
        lhs_points = lhs_sample(n_lhs, dim, rng)
        for x0 in lhs_points:
            if call_count[0] >= budget or time.time() > deadline:
                break
            counting_func(x0)

        # Phase 2: Multi-start local optimization from best LHS points
        # Sort by function value and pick top candidates for CMA-ES
        if best_x[0] is not None:
            # Use the best few points as starting points for local refinement
            # We'll run a local CMA-ES-like search from the best point
            # But since we can't easily use scipy's CMA-ES directly with counting,
            # we implement a simple adaptive (1+1)-ES with success-rate adaptation

            # Actually, let's use scipy.optimize.minimize with Nelder-Mead or COBYLA
            # from the best point, but we need to track calls carefully.
            # Instead, let's do a structured multi-start with adaptive step sizes.

            # Phase 2: Adaptive (1+1)-ES from the best initial point
            x = best_x[0].copy()
            fx = best_f[0]
            sigma = 1.0  # initial step size
            success_window = []
            window_size = 15

            remaining = budget - call_count[0]
            for i in range(remaining):
                if time.time() > deadline:
                    break

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = counting_func(y)

                success = fy < fx
                success_window.append(success)
                if len(success_window) > window_size:
                    success_window.pop(0)

                if success:
                    x, fx = y, fy
                else:
                    # Keep x
                    pass

                # Adaptive sigma based on success rate in window
                if len(success_window) >= window_size:
                    sr = np.mean(success_window)
                    # Target success rate ~ 1/5
                    if sr > 0.2:
                        sigma *= 1.05
                    elif sr < 0.2:
                        sigma *= 0.95
                    # Clamp sigma
                    sigma = np.clip(sigma, 0.01, 5.0)

            # Phase 3: If budget remains, try a few more random perturbations
            # around the best point with smaller steps
            remaining = budget - call_count[0]
            if remaining > 0:
                sigma_small = 0.1
                for i in range(remaining):
                    if time.time() > deadline:
                        break
                    y = np.clip(x + sigma_small * rng.standard_normal(dim), -5.0, 5.0)
                    fy = counting_func(y)
                    if fy < fx:
                        x, fx = y, fy
                    # Occasionally increase sigma for exploration
                    if i % 20 == 19:
                        sigma_small = min(sigma_small * 1.5, 1.0)

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
