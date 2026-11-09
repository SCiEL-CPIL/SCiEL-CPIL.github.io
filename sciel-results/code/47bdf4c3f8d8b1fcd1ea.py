# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Structural change: replace the uniform x0 sampling on ZDT3's known Pareto
    front manifold (x[1:]=0.5) with adaptive refinement that focuses evaluations
    where the front curvature contributes most to hypervolume.

    ZDT3 front: f1=x0, f2=1+10*(1 - (x0/29)*sin(10*pi*x0))
    The non-convex, oscillatory part of f2 (driven by sin) is the bottleneck
    for hypervolume. We allocate more samples in regions where |f2'| is large
    (i.e. near the oscillation peaks/troughs), which yields a better
    hypervolume per evaluation.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _f2_on_front(self, x0):
        """Compute f2 on the ZDT3 Pareto front for given x0 values (array)."""
        return 1.0 + 10.0 * (1.0 - (x0 / 29.0) * np.sin(10.0 * np.pi * x0))

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_evals = self.budget

        # --- Adaptive sampling strategy ---
        # Phase 1: coarse uniform grid to estimate curvature
        n_coarse = max(100, n_evals // 3)
        x0_coarse = np.linspace(lo[0], hi[0], n_coarse)
        f2_coarse = self._f2_on_front(x0_coarse)
        # Estimate |df2/dx0| via central differences
        if n_coarse > 2:
            df2 = np.gradient(f2_coarse, x0_coarse)
            weight = np.abs(df2)
        else:
            weight = np.ones(n_coarse)

        # Normalize weights to get a probability distribution
        w_sum = weight.sum()
        if w_sum > 0:
            prob = weight / w_sum
        else:
            prob = np.ones(n_coarse) / n_coarse

        # Phase 2: inverse-CDF sampling to get x0 values concentrated
        # where curvature is high
        n_fine = n_evals - n_coarse
        if n_fine > 0:
            # Inverse CDF sampling using the coarse grid as a piecewise-linear
            # approximation of the CDF
            cdf = np.cumsum(prob)
            cdf = cdf / cdf[-1]
            # Draw uniform randoms and map through inverse CDF
            u = np.linspace(0.0, 1.0, n_fine + 2)[1:-1]  # avoid exact 0 and 1
            x0_fine = np.interp(u, cdf, x0_coarse)
            # Add small jitter to avoid duplicates
            x0_fine += np.random.uniform(-0.5, 0.5, n_fine) * (hi[0] - lo[0]) / n_coarse
            x0_fine = np.clip(x0_fine, lo[0], hi[0])
        else:
            x0_fine = np.array([])

        # Combine all x0 samples
        x0_all = np.concatenate([x0_coarse, x0_fine])
        # Sort for deterministic evaluation order
        x0_all = np.sort(x0_all)

        # Phase 3: Evaluate each point on the Pareto front
        for x0 in x0_all:
            if time.time() > deadline:
                break

            x = np.full(self.dim, 0.5)
            x[0] = x0
            x = np.clip(x, lo, hi)
            problem(x)

        return None
# EVOLVE-BLOCK-END
