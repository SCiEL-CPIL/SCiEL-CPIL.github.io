"""Deterministic LHS-based multi-objective optimizer for WFG9."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Latin Hypercube Sampling based optimizer for WFG9 (3 objectives, 12 dims)."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        n = self.budget
        dim = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Strategy: Hybrid of Corner Sampling and LHS
        # Corners capture extreme objective values, LHS ensures uniform coverage.
        
        # 10% of budget for corners, 90% for LHS
        n_corners = max(10, n // 10)
        n_lhs = n - n_corners

        samples = []

        # Generate Corner Samples
        # Randomly select min or max for each dimension
        # Use a fixed seed for determinism
        rng = np.random.RandomState(42)
        for _ in range(n_corners):
            # 0 or 1 for each dimension
            corner_mask = rng.randint(0, 2, size=dim)
            # Map to lo or hi
            # If mask is 0 -> lo, if 1 -> hi
            # We can interpolate slightly inside to avoid numerical issues if bounds are tight, 
            # but problem() clips, so exact bounds are fine.
            point = np.where(corner_mask == 0, lo, hi)
            samples.append(point)

        # Generate LHS Samples
        if n_lhs > 0:
            sampler = qmc.LatinHypercube(d=dim, seed=123)
            lhs_samples_unit = sampler.random(n=n_lhs)
            lhs_samples = lo + (hi - lo) * lhs_samples_unit
            samples.append(lhs_samples)

        # Combine all samples
        X = np.vstack(samples)

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        return X
# EVOLVE-BLOCK-END
