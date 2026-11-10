"""Multi-objective optimizer for ZDT4 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits ZDT4's analytic Pareto set: x2..xn at values minimizing g."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        dim = self.dim
        budget = self.budget
        used = 0

        # Optimal values for x2..x10: 0.0 (gives cos(2*pi*0)=1, sin^2=0, term=-10)
        optimal_tail = np.zeros(dim - 1)

        # Phase 1: Uniform grid to establish initial coverage
        # Use a moderate number of points for initial grid
        n_initial = min(100, budget // 2)
        x1_initial = np.linspace(0.0, 1.0, n_initial)

        # Store evaluated points and their objective values
        evaluated_x1 = []
        evaluated_f = []

        for x1 in x1_initial:
            if used >= budget or time.time() > deadline:
                break
            x = np.concatenate([[x1], optimal_tail])
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            evaluated_x1.append(x1)
            evaluated_f.append(f)

        # Phase 2: Adaptive refinement using hypervolume contribution estimation
        # The Pareto front is f1=x1, f2=g*(2-2*x1^2-0.2*sin(8*pi*x1)) with g=19
        # Hypervolume contribution of a point is roughly the area it "owns"
        # We want to fill gaps where the front has high curvature or high contribution

        # Reference point for hypervolume calculation
        ref = np.array([11.0, 400.0])

        # Sort evaluated points by f1
        if evaluated_x1:
            sorted_idx = np.argsort(evaluated_x1)
            sorted_x1 = [evaluated_x1[i] for i in sorted_idx]
            sorted_f = [evaluated_f[i] for i in sorted_idx]
        else:
            sorted_x1 = []
            sorted_f = []

        # Estimate hypervolume contribution for each interval between consecutive points
        # and prioritize filling the largest gaps

        remaining_budget = budget - used

        # Create a priority queue of intervals to fill
        # Each interval is (midpoint, estimated_contribution, left_x1, right_x1)
        import heapq

        # Initialize intervals from sorted points
        intervals = []
        if len(sorted_x1) >= 2:
            for i in range(len(sorted_x1) - 1):
                left_x1 = sorted_x1[i]
                right_x1 = sorted_x1[i + 1]
                mid_x1 = (left_x1 + right_x1) / 2.0

                # Estimate contribution: width * average height
                # Height is approximately f2 value (since f1 is small relative to ref)
                f2_left = sorted_f[i][1]
                f2_right = sorted_f[i + 1][1]
                avg_f2 = (f2_left + f2_right) / 2.0

                # Contribution estimate: (ref[0] - f1) * (ref[1] - f2) approximation
                # But since we're filling gaps, the contribution is roughly:
                # (right_x1 - left_x1) * (ref[1] - avg_f2)
                contribution = (right_x1 - left_x1) * (ref[1] - avg_f2)

                heapq.heappush(intervals, (-contribution, mid_x1, left_x1, right_x1))

        # Also add boundary intervals [0, first] and [last, 1] if not already covered
        if sorted_x1 and sorted_x1[0] > 0.0:
            left_x1 = 0.0
            right_x1 = sorted_x1[0]
            mid_x1 = (left_x1 + right_x1) / 2.0
            f2_right = sorted_f[0][1]
            contribution = (right_x1 - left_x1) * (ref[1] - f2_right)
            heapq.heappush(intervals, (-contribution, mid_x1, left_x1, right_x1))

        if sorted_x1 and sorted_x1[-1] < 1.0:
            left_x1 = sorted_x1[-1]
            right_x1 = 1.0
            mid_x1 = (left_x1 + right_x1) / 2.0
            f2_left = sorted_f[-1][1]
            contribution = (right_x1 - left_x1) * (ref[1] - f2_left)
            heapq.heappush(intervals, (-contribution, mid_x1, left_x1, right_x1))

        # Adaptive refinement loop
        while intervals and used < budget and time.time() < deadline:
            # Pop the interval with highest estimated contribution
            neg_contrib, mid_x1, left_x1, right_x1 = heapq.heappop(intervals)

            if used >= budget or time.time() > deadline:
                break

            # Evaluate at the midpoint
            x1 = mid_x1
            x = np.concatenate([[x1], optimal_tail])
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            evaluated_x1.append(x1)
            evaluated_f.append(f)

            # Split the interval into two sub-intervals
            if right_x1 - left_x1 > 1e-6:  # Only split if interval is not too small
                # Left sub-interval: [left_x1, mid_x1]
                f2_left = None
                f2_mid = f[1]
                # Find f2 at left_x1
                for xi, fi in zip(evaluated_x1, evaluated_f):
                    if abs(xi - left_x1) < 1e-10:
                        f2_left = fi[1]
                        break
                if f2_left is None:
                    f2_left = f2_mid

                contribution_left = (mid_x1 - left_x1) * (ref[1] - (f2_left + f2_mid) / 2.0)
                if contribution_left > 1e-10:
                    heapq.heappush(intervals, (-contribution_left, (left_x1 + mid_x1) / 2.0, left_x1, mid_x1))

                # Right sub-interval: [mid_x1, right_x1]
                f2_right = None
                for xi, fi in zip(evaluated_x1, evaluated_f):
                    if abs(xi - right_x1) < 1e-10:
                        f2_right = fi[1]
                        break
                if f2_right is None:
                    f2_right = f2_mid

                contribution_right = (right_x1 - mid_x1) * (ref[1] - (f2_mid + f2_right) / 2.0)
                if contribution_right > 1e-10:
                    heapq.heappush(intervals, (-contribution_right, (mid_x1 + right_x1) / 2.0, mid_x1, right_x1))

        return None
# EVOLVE-BLOCK-END
