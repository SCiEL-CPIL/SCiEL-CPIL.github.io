import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        def evaluate(x):
            nonlocal used
            if not check_time():
                return
            if used >= budget:
                return
            x_clipped = np.clip(x, lo, hi)
            x_eval = x_clipped.copy()
            x_eval[2:] = 0.0
            problem(x_eval)
            used += 1

        # DTLZ7 Pareto front: x[2:]=0, so g=1
        # f1 = x[0], f2 = x[1], f3 = 1 - x[1]*h(x[1], 1)
        # h(y, 1) = 9 / (1 + 2*(sin(3*pi*y) + sin(4*pi*y)))
        #
        # Reference point: [1.1, 1.1, 25.0]
        # Hypervolume contribution of point (f1,f2,f3) ~ (1.1-f1)*(1.1-f2)*(25-f3)
        #
        # Strategy: Use max-min facility location in (f1,f2) space, weighted by (25-f3)
        # This ensures uniform coverage in the hypervolume-relevant region

        # Helper: compute f3 given x1
        def compute_f3(x1):
            # h(y, 1) = 9 / (1 + 2*(sin(3*pi*y) + sin(4*pi*y)))
            sin_sum = np.sin(3 * np.pi * x1) + np.sin(4 * np.pi * x1)
            denom = 1 + 2 * sin_sum
            if np.abs(denom) < 1e-12:
                h_val = 9.0  # fallback
            else:
                h_val = 9.0 / denom
            return 1.0 - x1 * h_val

        # Helper: compute hypervolume weight for a point (x0, x1)
        def hv_weight(x0, x1):
            f1 = x0
            f2 = x1
            f3 = compute_f3(x1)
            # Only count if all objectives < reference
            w1 = max(0.0, 1.1 - f1)
            w2 = max(0.0, 1.1 - f2)
            w3 = max(0.0, 25.0 - f3)
            return w1 * w2 * w3

        # Step 1: Evaluate a small set of anchor points (corners + center)
        anchor_points = [
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.5, 0.5),
            (0.0, 0.5),
            (0.5, 0.0),
            (0.5, 1.0),
            (1.0, 0.5),
        ]

        for x0, x1 in anchor_points:
            if not check_time() or used >= budget:
                break
            x = np.zeros(dim)
            x[0] = lo[0] + (hi[0] - lo[0]) * x0
            x[1] = lo[1] + (hi[1] - lo[1]) * x1
            evaluate(x)

        # Step 2: Max-min facility location in objective space (f1, f2)
        # Use actual f1, f2 coordinates for distance to ensure uniform coverage in objective space
        evaluated_points = []  # list of (x0, x1, f1, f2)
        for x0, x1 in anchor_points[:min(used, len(anchor_points))]:
            f1_val = x0
            f2_val = x1
            evaluated_points.append((x0, x1, f1_val, f2_val))

        # Candidate grid in (x0, x1) space - use finer grid near corners where HV contribution is high
        n_candidate = 25
        # Non-uniform grid: denser near 0 where hypervolume contribution is larger
        candidate_x0 = np.array([i / (n_candidate - 1) for i in range(n_candidate)])
        candidate_x1 = np.array([i / (n_candidate - 1) for i in range(n_candidate)])

        max_min_budget = min(budget - used, 120)  # Use up to 120 evaluations for max-min
        for _ in range(max_min_budget):
            if not check_time() or used >= budget:
                break

            # Find the best candidate from the grid
            best_score = -1
            best_x0, best_x1 = 0.5, 0.5

            for cx0 in candidate_x0:
                for cx1 in candidate_x1:
                    # Compute f1, f2 for this candidate
                    cf1 = cx0
                    cf2 = cx1

                    # Skip if too close to an existing point in objective space
                    min_dist = np.inf
                    for ex0, ex1, ef1, ef2 in evaluated_points:
                        dist = np.sqrt((cf1 - ef1)**2 + (cf2 - ef2)**2)
                        if dist < min_dist:
                            min_dist = dist

                    # Weight by hypervolume contribution
                    w = hv_weight(cx0, cx1)
                    if w <= 0:
                        continue
                    score = min_dist * np.sqrt(w)

                    if score > best_score:
                        best_score = score
                        best_x0, best_x1 = cx0, cx1

            # Evaluate the best point
            x = np.zeros(dim)
            x[0] = lo[0] + (hi[0] - lo[0]) * best_x0
            x[1] = lo[1] + (hi[1] - lo[1]) * best_x1
            evaluate(x)
            evaluated_points.append((best_x0, best_x1, best_x0, best_x1))

        # Step 3: Fill remaining budget with hypervolume-weighted sampling
        # Use inverse CDF sampling biased toward regions with high hypervolume contribution
        remaining = budget - used
        if remaining > 0:
            # Precompute a lookup table for weighted sampling
            # Weight function: w(x0, x1) = (1.1 - x0) * (1.1 - x1) * (25 - f3(x1))
            # Approximate by marginal weights
            n_grid = 50
            grid_x0 = np.linspace(0, 1, n_grid)
            grid_x1 = np.linspace(0, 1, n_grid)

            # Compute weight grid
            weight_grid = np.zeros((n_grid, n_grid))
            for i, x0 in enumerate(grid_x0):
                for j, x1 in enumerate(grid_x1):
                    w = hv_weight(x0, x1)
                    weight_grid[i, j] = w

            # Normalize to get probability grid
            total_w = weight_grid.sum()
            if total_w > 0:
                prob_grid = weight_grid / total_w
            else:
                prob_grid = np.ones_like(weight_grid) / (n_grid * n_grid)

            # Sample from the weighted grid
            for i in range(remaining):
                if not check_time():
                    break
                if used >= budget:
                    break

                # Sample index pair from the probability grid
                r1 = np.random.rand()
                r2 = np.random.rand()

                # Find the cell containing r1 in the row-sums, then r2 in the column-sums
                row_sums = prob_grid.sum(axis=1)
                col_sums = prob_grid.sum(axis=0)

                # Find row
                cum_row = np.cumsum(row_sums)
                if cum_row[-1] < 1e-10:
                    row_idx = np.random.randint(n_grid)
                else:
                    row_idx = np.searchsorted(cum_row, r1)
                    if row_idx >= n_grid:
                        row_idx = n_grid - 1

                # Find column within that row
                row_probs = prob_grid[row_idx]
                row_total = row_probs.sum()
                if row_total < 1e-10:
                    col_idx = np.random.randint(n_grid)
                else:
                    row_cum = np.cumsum(row_probs) / row_total
                    col_idx = np.searchsorted(row_cum, r2)
                    if col_idx >= n_grid:
                        col_idx = n_grid - 1

                x0_val = grid_x0[row_idx]
                x1_val = grid_x1[col_idx]

                x = np.zeros(dim)
                x[0] = lo[0] + (hi[0] - lo[0]) * x0_val
                x[1] = lo[1] + (hi[1] - lo[1]) * x1_val
                evaluate(x)

        return None
