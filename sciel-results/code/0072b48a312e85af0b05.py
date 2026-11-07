"""Deterministic global optimization with Sobol sequence and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sobol-based global search with adaptive local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        # Use Sobol sequence for low-discrepancy sampling
        # We'll implement a simple Sobol-like sequence using Halton sequence
        # which is easier to implement without scipy

        def halton_sequence(index, dim):
            """Generate a Halton sequence point at the given index."""
            point = np.zeros(dim)
            for i in range(dim):
                base = i + 2  # Primes: 2, 3, 5, 7, 11, ...
                f = 1.0
                r = 0.0
                n = index
                while n > 0:
                    f /= base
                    r += f * (n % base)
                    n //= base
                point[i] = r
            return point

        def map_to_box(point, lower=-5.0, upper=5.0):
            """Map [0,1]^dim to [lower,upper]^dim."""
            return lower + (upper - lower) * point

        # Phase 1: Global exploration with Sobol/Halton sequence
        # Use first ~400 evaluations for global search
        n_global = 400

        best_x = None
        best_f = np.inf
        best_index = -1

        # Evaluate Halton points
        for i in range(1, n_global + 1):
            if time.time() > deadline:
                break

            h = halton_sequence(i, self.dim)
            x = map_to_box(h)
            x = np.clip(x, -5, 5)
            fx = func(x)

            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                best_index = i

        # Phase 2: Local refinement around the best point
        # Use remaining budget for local search
        remaining = self.budget - n_global
        if remaining <= 0:
            return best_x, best_f

        # Adaptive local search with decreasing step size
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        rng = np.random.default_rng(42)
        sigma = 1.0  # Initial step size

        for i in range(remaining):
            if time.time() > deadline:
                break

            # Generate candidate
            y = best_x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            fy = func(y)

            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                sigma *= 1.2  # Expand slightly on success
            else:
                sigma *= 0.85  # Shrink on failure

            # If sigma becomes very small, try a larger jump
            if sigma < 0.01:
                # Occasionally make a larger exploratory jump
                if rng.random() < 0.1:
                    sigma = 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
