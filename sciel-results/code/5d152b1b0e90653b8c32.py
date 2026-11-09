"""Multi-objective optimizer on RE34 using deterministic Latin Hypercube Sampling."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic LHS-based multi-objective search for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Phase 1: Deterministic Latin Hypercube Sampling for full coverage
        # Use all budget evaluations as LHS points for maximum coverage
        n_lhs = budget
        # Generate LHS: for each dimension, partition [0,1] into n equal intervals,
        # pick one random point in each interval, then permute across dimensions.
        # Use a fixed seed for determinism.
        rng = np.random.default_rng(42)
        lhs = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Create intervals [i/n, (i+1)/n) for i in range(n)
            intervals = np.arange(n_lhs) / n_lhs
            # Pick a uniform point within each interval
            offsets = rng.random(n_lhs)
            lhs[:, d] = intervals + offsets * (1.0 / n_lhs)
            # Permute the assignments for this dimension
            lhs[:, d] = lhs[rng.permutation(n_lhs), d]

        # Map from [0,1]^dim to [lo, hi]^dim
        X_lhs = lo + lhs * span
        X_lhs = np.clip(X_lhs, lo, hi)

        # Evaluate all LHS points
        F_lhs = np.array([problem(x) for x in X_lhs])

        # Filter out non-finite or out-of-bounds results (shouldn't happen due to clipping)
        valid_mask = np.all(np.isfinite(F_lhs), axis=1)
        if not np.all(valid_mask):
            F_lhs = F_lhs[valid_mask]
            X_lhs = X_lhs[valid_mask]

        # Return the LHS points (the return value is ignored by the evaluator,
        # but we keep it for interface consistency)
        return X_lhs
# EVOLVE-BLOCK-END
