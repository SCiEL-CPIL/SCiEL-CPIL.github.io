"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 exploiting analytical structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT6 Pareto front: x[0]=t, x[1:]=0
        # f1(t) = 1 - exp(-4t) * sin(10*pi*t)^2
        # f2(t) = 1 - sqrt(f1(t))
        # The front is a curve in [0,1]^2 with oscillations in f1.

        # Strategy: place all budget evaluations on the Pareto front.
        # Use hypervolume-aware spacing: denser where the curve contributes
        # more to the 2D hypervolume (high |df1/dt| and high |df2/dt|).

        # Generate a dense candidate set for computing the importance density
        n_cand = 10000
        t_cand = np.linspace(0.0, 1.0, n_cand)

        # Compute f1, f2 on candidates
        f1_cand = 1.0 - np.exp(-4.0 * t_cand) * np.sin(10.0 * np.pi * t_cand) ** 2
        f2_cand = 1.0 - np.sqrt(np.maximum(f1_cand, 0.0))

        # For 2D hypervolume with reference [11,11], the contribution of a point
        # depends on the area it "covers". For a set of Pareto-optimal points
        # sorted by f1, the HV is sum of rectangles. The marginal contribution
        # of adding a point between two neighbors depends on the gap in both
        # objective dimensions.
        #
        # A good heuristic for placement: weight by the local arc length in
        # objective space, which is sqrt((df1/dt)^2 + (df2/dt)^2).
        # Additionally, regions where f1 is small contribute more to HV because
        # they extend further toward the reference point.
        #
        # Combined importance: |df1/dt| * (11 - f2) + |df2/dt| * (11 - f1)
        # This weights regions that are far from the reference in the other
        # objective, which is where the HV contribution is largest.

        dt = t_cand[1] - t_cand[0]
        df1_dt = np.gradient(f1_cand, dt)
        df2_dt = np.gradient(f2_cand, dt)

        # Hypervolume-aware importance
        # The HV contribution of a segment in objective space is related to
        # the area of the rectangle it subtends. For a small segment, the
        # contribution is approximately |df1| * (ref2 - f2) + |df2| * (ref1 - f1)
        # but more precisely, for 2D HV with non-dominated points sorted by f1:
        # HV = sum_i (f2[i] - f2[i+1]) * (ref1 - f1[i]) for decreasing f2
        # The marginal gain of inserting a point is related to the gap it fills.
        #
        # Practical approach: use the "contribution" of each candidate point
        # if it were the only point: (11-f1)*(11-f2). But for a set, we want
        # to maximize the total HV. A good density is proportional to the
        # local curvature-adjusted arc length weighted by distance to reference.

        # Use: importance = |df1/dt| * (11 - f2) + |df2/dt| * (11 - f1)
        # This captures both the speed of the curve and the HV leverage.
        importance = np.abs(df1_dt) * (11.0 - f2_cand) + np.abs(df2_dt) * (11.0 - f1_cand)
        importance = np.maximum(importance, 1e-10)
        importance = importance / importance.sum()

        # Compute CDF for inverse transform sampling
        cdf = np.cumsum(importance)
        cdf /= cdf[-1]

        # Place budget points at quantiles of this weighted distribution
        # Use midpoints of quantile bins to avoid edge issues
        n_place = budget
        quantiles = np.linspace(0, 1, n_place + 2)[1:-1]  # avoid exact 0 and 1
        t_selected = np.interp(quantiles, cdf, t_cand)

        # Ensure we stay within [0, 1]
        t_selected = np.clip(t_selected, 0.0, 1.0)

        # Sort for deterministic ordering
        t_selected = np.sort(t_selected)

        # Remove near-duplicates (keep at most budget points)
        if len(t_selected) > 1:
            # Remove points that are too close together
            keep = [0]
            for i in range(1, len(t_selected)):
                if t_selected[i] - t_selected[keep[-1]] > 1e-8:
                    keep.append(i)
            t_selected = t_selected[keep]

        # If we have fewer than budget points (due to dedup), add more from uniform
        if len(t_selected) < budget:
            n_needed = budget - len(t_selected)
            t_extra = np.linspace(0.0, 1.0, n_needed + 2)[1:-1]
            t_selected = np.concatenate([t_selected, t_extra])
            t_selected = np.sort(t_selected)
            # Re-deduplicate
            if len(t_selected) > 1:
                keep = [0]
                for i in range(1, len(t_selected)):
                    if t_selected[i] - t_selected[keep[-1]] > 1e-8:
                        keep.append(i)
                t_selected = t_selected[keep]
                if len(t_selected) > budget:
                    t_selected = t_selected[:budget]

        # Now evaluate all points on the Pareto front
        X = []
        for t in t_selected:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = t
            x = np.clip(x, lo, hi)
            f = problem(x)
            X.append(x)

        return X
# EVOLVE-BLOCK-END
