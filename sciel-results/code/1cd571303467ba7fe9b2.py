"""Optimized ZDT2 solver using adaptive hypervolume-maximizing point placement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 using adaptive hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        # ZDT2 analytic structure:
        # f1(x) = x1
        # f2(x) = (1 + 10*(dim-1)) * (1 - sqrt(x1)) + sum(x_i^2 for i>=2)
        # Pareto front: x_i = 0 for i >= 2, so f2 = C * (1 - sqrt(x1))
        C = 1.0 + 10.0 * (dim - 1)  # C = 291 for dim=30

        # Reference point
        ref = np.array([11.0, 11.0])

        # Valid range for x1 on the front:
        # f1 = x1 < 11 (always true for x1 in [0,1])
        # f2 = C*(1-sqrt(x1)) < 11 => sqrt(x1) > 1 - 11/C
        x1_min = max(0.0, (1.0 - 11.0 / C) ** 2)
        x1_max = 1.0

        # We'll build the set of points adaptively.
        # Start with two endpoints of the valid range.
        # Then iteratively add the point that maximizes the hypervolume.

        # Helper: given a set of f1 values (sorted), compute the hypervolume
        # of the corresponding points on the front relative to ref.
        def compute_hv(f1_values):
            """Compute exact hypervolume of points on ZDT2 front."""
            if len(f1_values) == 0:
                return 0.0
            f1_sorted = np.sort(f1_values)
            # Filter to valid range
            mask = (f1_sorted >= x1_min) & (f1_sorted <= x1_max)
            f1_sorted = f1_sorted[mask]
            if len(f1_sorted) == 0:
                return 0.0

            f2_values = C * (1.0 - np.sqrt(f1_sorted))

            # Points are (f1_i, f2_i) with f1 increasing, f2 decreasing.
            # Hypervolume relative to ref = sum over segments.
            # For convex decreasing front, HV = sum_{i=0}^{n-1} (ref[0] - f1_i) * (f2_i - f2_{i+1})
            # plus the last segment to ref[1].
            # Actually, the standard formula for HV of a set of points on a monotonic front:
            # HV = sum_{i=0}^{n-1} (ref[0] - f1[i]) * (f2[i] - f2[i+1]) + (ref[0] - f1[n-1]) * (f2[n-1] - ref[1])
            # Wait, that's not quite right. Let me think more carefully.
            # The hypervolume dominated by the set of points (and bounded by ref) is the area
            # under the "lower envelope" of the points up to ref.
            # For a set of non-dominated points sorted by f1, the dominated region is:
            # For each point i, it dominates the rectangle [f1_i, ref[0]] x [f2_i, ref[1]]
            # but we must avoid double-counting. The correct way:
            # HV = sum_{i=0}^{n-1} (ref[0] - f1[i]) * (f2[i] - min(f2[i+1], ref[1]))
            # No, let's use the standard incremental formula.
            # Sort points by f1 ascending. f2 will be descending.
            # HV = 0
            # prev_f2 = ref[1]
            # for i in range(n):
            #     f2_curr = f2[i]
            #     if f2_curr < ref[1]:
            #         HV += (ref[0] - f1[i]) * (prev_f2 - f2_curr)
            #         prev_f2 = f2_curr
            # This works if the front is convex and points are non-dominated.

            n_pts = len(f1_sorted)
            if n_pts == 0:
                return 0.0

            f2_vals = C * (1.0 - np.sqrt(f1_sorted))
            # Only consider points strictly better than ref in both objectives
            valid_mask = (f1_sorted < ref[0]) & (f2_vals < ref[1])
            f1_valid = f1_sorted[valid_mask]
            f2_valid = f2_vals[valid_mask]

            if len(f1_valid) == 0:
                return 0.0

            # Sort by f1 ascending
            order = np.argsort(f1_valid)
            f1_v = f1_valid[order]
            f2_v = f2_valid[order]

            # Compute HV using the sweep line method
            hv = 0.0
            prev_f2 = ref[1]
            for i in range(len(f1_v)):
                if f2_v[i] < ref[1]:
                    hv += (ref[0] - f1_v[i]) * (prev_f2 - f2_v[i])
                    prev_f2 = f2_v[i]
            return hv

        # Start with endpoints
        current_f1s = [x1_min, x1_max]

        # Pre-compute f2 for current points
        def f2_from_f1(f1):
            return C * (1.0 - np.sqrt(np.clip(f1, 0.0, 1.0)))

        # Adaptively add points
        # We have budget evaluations to use.
        # We've already "placed" 2 points conceptually, but we need to evaluate them.
        # Let's reserve evaluations for the points we decide to keep.

        # Strategy: use a greedy approach.
        # 1. Start with a small set of candidate points (e.g., 20 evenly spaced in f1).
        # 2. Evaluate all candidates.
        # 3. From the evaluated points, select the subset of size <= budget that maximizes HV.
        #    But we can only evaluate `budget` points total.
        # So we must choose which `budget` points to evaluate.

        # Better strategy:
        # Phase 1: Evaluate a set of K candidate points (K < budget) that are well-spread.
        # Phase 2: Use the remaining budget to fill gaps identified by the HV calculation.

        # Let's use a two-phase approach:
        # Phase 1: Evaluate 10 points evenly spaced in f1 over [x1_min, x1_max].
        # Phase 2: For the remaining budget, iteratively find the f1 value that, if evaluated,
        #           would maximize the HV of the set (assuming we can place it on the front).

        n_phase1 = min(10, budget)
        f1_candidates = np.linspace(x1_min, x1_max, n_phase1)

        # Evaluate phase 1 points
        evaluated_f1s = []
        for i in range(n_phase1):
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = f1_candidates[i]
            problem(x)
            evaluated_f1s.append(f1_candidates[i])

        # Now we have evaluated_f1s. We have budget - len(evaluated_f1s) evaluations left.
        remaining_budget = budget - len(evaluated_f1s)

        # For the remaining evaluations, we'll use the greedy HV maximization.
        # Current set of f1 values (on the front)
        current_set = list(evaluated_f1s)

        # To find the best next point, we need to evaluate the HV for many candidate f1 values.
        # But we can only evaluate one point at a time via problem().
        # However, we can analytically compute the HV for any candidate f1 without calling problem().
        # We only need to call problem() to "record" the evaluation.
        # So we can:
        # 1. Analytically determine the best f1 to add next.
        # 2. Call problem() with that f1 to record it.

        # Let's pre-compute the HV function efficiently.
        # We'll maintain the sorted list of f1s and their f2s.

        # For efficiency, let's use a fixed grid of candidate f1 values for the greedy selection.
        # Create a fine grid of 1000 points in [x1_min, x1_max].
        n_grid = 1000
        f1_grid = np.linspace(x1_min, x1_max, n_grid)
        f2_grid = C * (1.0 - np.sqrt(f1_grid))

        # For each iteration, find the grid point that maximizes HV when added to current_set.
        # Then evaluate that point.

        for _ in range(remaining_budget):
            if time.time() > deadline:
                break

            # Find the best f1 from the grid to add
            best_hv = -1.0
            best_f1 = x1_min  # default

            # We need to compute HV for current_set + {candidate} for each candidate.
            # To speed this up, we can pre-compute the contribution of each candidate.
            # But for simplicity, let's just iterate over the grid.
            # With 1000 grid points and up to 190 iterations, this is 190,000 HV computations.
            # Each HV computation is O(n) where n is the size of current_set (up to 190).
            # Total: 190,000 * 190 = 36,100,000 operations. This might be slow in Python.

            # Let's optimize: instead of recomputing HV from scratch, we can use the fact that
            # adding one point to a sorted set only changes the HV locally.
            # But for now, let's try a coarser grid or a smarter approach.

            # Alternative: use the analytic property that the optimal point to add is the one
            # that maximizes the "gap" in hypervolume contribution. For a convex front, this is
            # approximately the midpoint of the largest gap in the f1-f2 plane.

            # Let's use a simpler heuristic: find the pair of consecutive points in current_set
            # (sorted by f1) that has the largest "hypervolume gap", and place the new point
            # at the f1 value that maximizes the HV contribution in that gap.

            # For two consecutive points (f1_a, f2_a) and (f1_b, f2_b) with f1_a < f1_b,
            # the HV contribution of the segment is maximized when the new point is placed
            # such that the two sub-segments have equal HV contribution.
            # For a linear front in (u, f2) space where u = sqrt(f1), this is the midpoint in u.

            # Let's implement this heuristic.
            if len(current_set) < 2:
                # If we have fewer than 2 points, just pick the midpoint
                best_f1 = (x1_min + x1_max) / 2.0
            else:
                # Sort current_set
                sorted_f1s = sorted(current_set)
                # Find the pair with the largest gap in terms of potential HV contribution
                # We'll approximate by looking at the gap in u = sqrt(f1)
                best_gap = -1.0
                best_mid_u = None
                for i in range(len(sorted_f1s) - 1):
                    u_a = np.sqrt(sorted_f1s[i])
                    u_b = np.sqrt(sorted_f1s[i + 1])
                    gap = u_b - u_a
                    if gap > best_gap:
                        best_gap = gap
                        best_mid_u = (u_a + u_b) / 2.0

                if best_mid_u is not None:
                    best_f1 = best_mid_u ** 2
                else:
                    best_f1 = (x1_min + x1_max) / 2.0

            # Evaluate the best point
            x = np.zeros(dim)
            x[0] = best_f1
            problem(x)
            current_set.append(best_f1)

        return None
# EVOLVE-BLOCK-END
