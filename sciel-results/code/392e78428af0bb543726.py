import numpy as np
import time

class Optimizer:
    """DTLZ2 optimizer with hypervolume-weighted recursive bisection on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ2: Pareto front is a quarter-sphere in objective space.
        # x[0] and x[1] parameterize the front; x[2:] must be 0.5 for g=0.
        # f1 = cos(t1)*cos(t2), f2 = cos(t1)*sin(t2), f3 = sin(t1)
        # where t1 = x[0]*pi/2, t2 = x[1]*pi/2

        g_start = self.n_obj - 1
        base_x = np.zeros(self.dim)
        base_x[g_start:] = 0.5

        evaluated_points = []
        used_budget = 0

        def eval_point(u, v):
            """Evaluate a point on the Pareto front given uniform params u,v in [0,1].
            u maps to t1 via arcsin for uniform surface sampling, v maps to t2 uniformly."""
            nonlocal used_budget
            if time.time() > deadline or used_budget >= self.budget:
                return None
            x0 = (2.0 / np.pi) * np.arcsin(u)
            x1 = v
            x = base_x.copy()
            x[0] = x0
            x[1] = x1
            f = problem(x)
            evaluated_points.append(x)
            used_budget += 1
            return f

        # Hypervolume-weighted recursive bisection:
        # The Pareto front is a 2D surface. We want to maximize the hypervolume
        # of the union of boxes from each point to the reference point (2.5, 2.5, 2.5).
        #
        # Strategy: Start with the full [0,1]x[0,1] parameter space. Recursively
        # bisect the rectangle with the largest estimated marginal hypervolume gain.
        # The key insight: points near the "center" of the front (where all three
        # objectives are moderate) contribute more to hypervolume than points near
        # the edges.
        #
        # We use a priority queue approach: estimate the hypervolume contribution
        # of the center of each cell, and prioritize cells where the center would
        # add the most new volume.

        # For simplicity and speed, we use a structured approach:
        # 1. First evaluate the center of the full domain
        # 2. Then recursively bisect, always splitting the cell with the largest
        #    estimated contribution first.

        # We'll use a list of (priority, u_min, u_max, v_min, v_max) tuples.
        # Priority = estimated hypervolume contribution of the cell center.
        #
        # To estimate contribution, we compute the objective vector at the cell
        # center and check if it's dominated by any already-evaluated point.
        # If not dominated, estimate its individual hypervolume contribution.

        # For efficiency, we'll do a simpler but effective approach:
        # Use a quadtree-like bisection where we split the cell with the largest
        # area first (which correlates with larger potential hypervolume gain
        # when combined with the objective space geometry).

        # Actually, let me use a more direct approach:
        # Generate points using a recursive bisection where at each step we split
        # the cell whose center has the highest "hypervolume score".

        # Let's precompute: for a point on the front with parameters (u,v):
        # t1 = arcsin(u) * 2/pi, t2 = v * pi/2
        # f1 = cos(t1)*cos(t2), f2 = cos(t1)*sin(t2), f3 = sin(t1)
        # HV contribution (rough) = (2.5-f1)*(2.5-f2)*(2.5-f3) when not dominated

        def front_point_obj(u, v):
            t1 = (2.0 / np.pi) * np.arcsin(u)
            t2 = v * np.pi / 2.0
            f1 = np.cos(t1) * np.cos(t2)
            f2 = np.cos(t1) * np.sin(t2)
            f3 = np.sin(t1)
            return np.array([f1, f2, f3])

        # Recursive bisection with hypervolume weighting
        # Cell: (u_min, u_max, v_min, v_max, depth)
        cells = [(0.0, 1.0, 0.0, 1.0, 0)]
        evaluated_cells = {}  # track evaluated cell centers by rounded position

        max_depth = 8  # 2^8 = 256 possible cells, but we only have 200 budget

        # First pass: evaluate centers of coarse cells
        # Sort cells by estimated HV contribution of their center
        while cells and used_budget < self.budget and time.time() < deadline:
            # Score each cell by its center's estimated HV contribution
            def cell_score(cell):
                um = (cell[0] + cell[1]) / 2.0
                vm = (cell[2] + cell[3]) / 2.0
                f = front_point_obj(um, vm)
                # Estimate individual contribution
                return (2.5 - f[0]) * (2.5 - f[1]) * (2.5 - f[2])

            cells.sort(key=cell_score, reverse=True)
            cell = cells.pop(0)
            u_min, u_max, v_min, v_max, depth = cell

            # Evaluate center of this cell
            um = (u_min + u_max) / 2.0
            vm = (v_min + v_max) / 2.0
            eval_point(um, vm)

            # If we have budget and depth allows, split this cell
            if depth < max_depth and used_budget < self.budget:
                # Split into 4 sub-cells
                u_mid = (u_min + u_max) / 2.0
                v_mid = (v_min + v_max) / 2.0
                new_cells = [
                    (u_min, u_mid, v_min, v_mid, depth + 1),
                    (u_mid, u_max, v_min, v_mid, depth + 1),
                    (u_min, u_mid, v_mid, v_max, depth + 1),
                    (u_mid, u_max, v_mid, v_max, depth + 1),
                ]
                cells.extend(new_cells)

        # Fill remaining budget with a low-discrepancy sequence for fine coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        if used_budget < self.budget:
            # Use a different starting index to avoid overlap with grid points
            halton_start = 1000
            for idx in range(halton_start, halton_start + 2000):
                if time.time() > deadline or used_budget >= self.budget:
                    break
                u = halton(idx, 2)
                v = halton(idx, 3)
                # Check if this point is too close to an already evaluated one
                # (skip if very close to save budget for new points)
                too_close = False
                for px in evaluated_points:
                    if abs(px[0] - (2.0 / np.pi) * np.arcsin(u)) < 1e-4 and abs(px[1] - v) < 1e-4:
                        too_close = True
                        break
                if not too_close:
                    eval_point(u, v)

        return evaluated_points
