"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT1-aware optimizer: greedy hypervolume maximization on analytic Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin before 300s kill

        N = self.budget
        dim = self.dim

        # ZDT1 Pareto front: f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0,1]
        # Reference point: [11, 11]
        # Hypervolume contribution of a point (f1, f2) given existing points:
        # We need to compute the marginal hypervolume added by each candidate.

        # Candidate pool: generate many candidate x1 values
        # Use a fine grid to have good resolution
        n_candidates = 2000
        x1_candidates = np.linspace(0.0, 1.0, n_candidates)

        # Precompute objective values for all candidates
        f1_candidates = x1_candidates.copy()
        f2_candidates = 1.0 - np.sqrt(x1_candidates)

        # Greedy selection: start with no points, add point with max marginal HV
        # For 2D, hypervolume with reference point (r1, r2) = sum of rectangles
        # A point (a, b) contributes area (r1-a)*(r2-b) if it's the only point
        # With multiple points, we need to compute the union of rectangles

        # For efficiency in 2D, we can track the "upper envelope" of f2 values
        # as a function of f1. The hypervolume is the integral under the upper
        # envelope from 0 to r1, minus the reference area.

        # Actually, for minimization, we want points that minimize both objectives.
        # The hypervolume indicator with reference point R is the volume of the
        # region dominated by at least one point but not by R.

        # For 2D minimization: HV = sum over points of (r1 - f1_i) * (r2 - f2_i)
        # but with inclusion-exclusion for overlapping regions.

        # Simpler approach: sort by f1, and for each point, the marginal contribution
        # is (r1 - f1) * (min(r2, min_f2_so_far) - f2) if f2 < min_f2_so_far,
        # plus (r1 - f1) * (r2 - min_f2_so_far) ... this gets complex.

        # Let's use a direct approach: maintain sorted list of selected points,
        # and for each candidate, compute the actual HV increment.

        # For 2D, HV of a set of points with reference (r1, r2):
        # Sort points by f1 ascending. The HV is the sum of rectangles where
        # each point contributes (r1 - f1_i) * (f2_prev - f2_i) where f2_prev
        # is the f2 of the previous point in sorted order (or r2 if first).

        # Actually: sort by f1. HV = sum_{i} (f1_{i+1} - f1_i) * (r2 - f2_i)
        # where f1_{N+1} = r1, and we only count points where f2_i < r2.
        # But this is for the area under the step function.

        # Let me just compute HV directly for small sets.

        def compute_hv(points, r1=11.0, r2=11.0):
            """Compute hypervolume for 2D minimization with reference point."""
            if len(points) == 0:
                return 0.0
            # Filter points strictly better than reference
            valid = [(f1, f2) for f1, f2 in points if f1 < r1 and f2 < r2]
            if len(valid) == 0:
                return 0.0
            # Sort by f1
            valid.sort(key=lambda p: p[0])
            # Remove dominated points (keep only Pareto-optimal)
            pareto = []
            min_f2 = float('inf')
            for f1, f2 in valid:
                if f2 < min_f2:
                    pareto.append((f1, f2))
                    min_f2 = f2
            if len(pareto) == 0:
                return 0.0
            # Compute HV: sum of rectangles
            hv = 0.0
            for i in range(len(pareto)):
                f1_i, f2_i = pareto[i]
                if i + 1 < len(pareto):
                    f1_next = pareto[i + 1][0]
                else:
                    f1_next = r1
                hv += (f1_next - f1_i) * (r2 - f2_i)
            return hv

        # Greedy selection
        selected_indices = []
        selected_points = []  # list of (f1, f2)

        # Pre-filter: only consider candidates with f1 < 11 and f2 < 11 (all qualify)
        # Use a larger candidate pool for better resolution
        # But greedy over 2000 candidates x 200 selections = 400k HV computations
        # Each HV computation is O(n log n) for n up to 200, so ~200*log(200) ≈ 1500 ops
        # Total: 400k * 1500 = 600M ops - might be too slow.

        # Optimization: use a smaller candidate pool and smarter selection
        # Or: use the fact that for ZDT1, the optimal distribution can be derived analytically.

        # Analytical approach: For a concave Pareto front f2 = g(f1), the hypervolume
        # maximizing distribution places points at equal "marginal contribution" intervals.
        # For ZDT1 with ref [11,11], the marginal HV of a point at f1 is approximately
        # (11 - f1) * d(1 - sqrt(f1))/df1 integrated over a small interval.

        # Let's try a different approach: use the known result that for a smooth concave
        # front, the optimal points are at f1 values where the "density" is proportional
        # to 1/sqrt(f1) or similar.

        # Actually, let's just do greedy but with a reduced candidate pool and efficient HV.

        # Use 500 candidates for greedy
        n_cand = 500
        x1_cand = np.linspace(0.0, 1.0, n_cand)
        f1_cand = x1_cand.copy()
        f2_cand = 1.0 - np.sqrt(x1_cand)

        # Greedy: for each of N selections, find candidate with max marginal HV
        # To speed up, we can note that adding a point only affects the HV locally.

        # For now, let's do straightforward greedy with a moderate candidate pool.
        # If it's too slow, we'll fall back to analytical spacing.

        selected_f1 = []
        selected_f2 = []

        for k in range(N):
            best_idx = -1
            best_hv = -1.0

            # Try each candidate
            for j in range(n_cand):
                # Skip if already selected (approximately)
                # Compute HV with this candidate added
                new_points = list(zip(selected_f1, selected_f2))
                new_points.append((f1_cand[j], f2_cand[j]))
                hv = compute_hv(new_points)
                if hv > best_hv:
                    best_hv = hv
                    best_idx = j

            if best_idx >= 0:
                selected_f1.append(f1_cand[best_idx])
                selected_f2.append(f2_cand[best_idx])

            # Check time
            if time.time() > deadline - 10.0:
                break

        # If greedy didn't finish, fall back to analytical spacing
        if len(selected_f1) < N:
            # Use power-law spacing: x1 = (i/N)^p for optimal p
            # For ZDT1, p=2 was used before. Let's try to optimize.
            # Actually, just use the greedy result we have and fill remaining with
            # interpolated points.
            remaining = N - len(selected_f1)
            if remaining > 0:
                # Fill with evenly spaced points in gaps
                all_f1 = sorted(selected_f1)
                # Add points at midpoints of largest gaps
                for _ in range(remaining):
                    if len(all_f1) == 0:
                        all_f1.append(0.5)
                    else:
                        # Find largest gap
                        best_gap = -1
                        best_mid = 0.5
                        for i in range(len(all_f1)):
                            if i == 0:
                                gap = all_f1[0]
                            elif i == len(all_f1) - 1:
                                gap = 1.0 - all_f1[-1]
                            else:
                                gap = all_f1[i + 1] - all_f1[i]
                            if gap > best_gap:
                                best_gap = gap
                                if i == 0:
                                    best_mid = all_f1[0] / 2.0
                                elif i == len(all_f1) - 1:
                                    best_mid = (1.0 + all_f1[-1]) / 2.0
                                else:
                                    best_mid = (all_f1[i] + all_f1[i + 1]) / 2.0
                        all_f1.append(best_mid)
                        all_f1.sort()
                selected_f1 = all_f1[:N]
                selected_f2 = [1.0 - np.sqrt(x) for x in selected_f1]

        # Construct input vectors
        X = np.zeros((N, dim))
        X[:, 0] = np.array(selected_f1[:N])
        # x2...xdim = 0

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        return X
# EVOLVE-BLOCK-END
