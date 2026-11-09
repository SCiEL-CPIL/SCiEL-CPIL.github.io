# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MOEA/D style multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_3d(self, points, ref):
        """Compute exact hypervolume for a set of 3D points against a reference point."""
        if len(points) == 0:
            return 0.0
        mask = np.all(points < ref, axis=1)
        pts = points[mask]
        if len(pts) == 0:
            return 0.0

        # Sort by first objective
        order = np.argsort(pts[:, 0])
        pts = pts[order]

        hv = 0.0
        n = len(pts)
        for i in range(n):
            # For point i, compute the volume where it is the best in objective 0
            # and it dominates all previous points in objectives 1 and 2
            x0 = pts[i, 0]
            # Find the range of points j > i where pts[j,1] < pts[i,1] and pts[j,2] < pts[i,2]
            # Actually, standard 3D HV algorithm:
            # Sort by x. For each point, the contribution is (x[i] - x[i-1]) * area_in_2d
            # where area_in_2d is the union of rectangles formed by points with index >= i
            # that are dominated by point i in objectives 2 and 3.

            # Let's use a simpler O(n^3) approach for small n, or a better O(n^2 log n)
            pass

        # Better approach: use the standard recursive HV algorithm
        return self._hv_recursive(pts, ref)

    def _hv_recursive(self, pts, ref):
        """Recursive hypervolume computation."""
        n = len(pts)
        if n == 0:
            return 0.0
        if n == 1:
            return max(0, ref[0] - pts[0, 0]) * max(0, ref[1] - pts[0, 1]) * max(0, ref[2] - pts[0, 2])

        # Sort by first objective
        order = np.argsort(pts[:, 0])
        pts = pts[order]

        # The first point (smallest x) dominates all in x
        # Its contribution is (x[1] - x[0]) * HV_2d of the remaining points projected on (y, z)
        # Actually, we need to be careful. Let's use the standard algorithm.

        # Standard 3D HV algorithm:
        # 1. Sort points by x coordinate
        # 2. For each point i from left to right:
        #    - The volume contributed by point i is (x[i+1] - x[i]) * A_i
        #      where A_i is the 2D area of the union of rectangles (y_j, z_j) for all j >= i
        #      such that y_j < y_i and z_j < z_i... no, that's not right.

        # Let me use a cleaner formulation.
        # HV = sum over all points p of (p.x - p_prev.x) * Area2D(points dominated by p in y,z and with x >= p.x)
        # This is getting complex. Let me just use a brute force O(n^3) for n <= 200.

        # Brute force: for each point, compute the volume it uniquely dominates
        # A point p uniquely dominates the region [p.x, x_next) x [p.y, y_bound) x [p.z, z_bound)
        # where x_next is the x of the next point in sorted order
        # and y_bound, z_bound are determined by other points.

        # Actually, let me just implement the standard recursive algorithm properly.
        # Reference: "A Fast Hypervolume Algorithm for 3D"

        if n <= 1:
            if n == 0:
                return 0.0
            return float(max(0, ref[0] - pts[0, 0]) * max(0, ref[1] - pts[0, 1]) * max(0, ref[2] - pts[0, 2]))

        # Sort by x
        pts = pts[np.argsort(pts[:, 0])]

        # The leftmost point contributes a slab of width (x[1] - x[0]) times the 2D area
        # of the projection of all points with index >= 0 onto (y, z) plane, but only
        # the part dominated by point 0.

        # Actually, the correct recursive step:
        # HV(pts, ref) = (x[1] - x[0]) * HV_2d({(y_j, z_j) for j in 0..n-1}, (ref[1], ref[2]))
        #                + HV({pts[j] for j in 1..n-1}, ref)
        # where HV_2d is the 2D hypervolume of the projections.

        # Wait, that's not quite right either. Let me think again.

        # The correct decomposition:
        # Sort by x. The first point p0 has the smallest x.
        # The region [p0.x, p1.x) x [0, ref[1]) x [0, ref[2]) is partially covered.
        # Specifically, for x in [p0.x, p1.x), the cross-section is the 2D region dominated
        # by all points with x <= current_x. Since p0 has the smallest x, for x in [p0.x, p1.x),
        # only p0 is active. So the cross-section is the rectangle [0, p0.y) x [0, p0.z).
        # But we need the area NOT covered by other points... no, HV is the union of all boxes.

        # I'm overcomplicating this. Let me just use a well-known implementation.

        # For n up to 200, O(n^3) is fine: 8,000,000 operations max.
        return self._hv_bruteforce(pts, ref)

    def _hv_bruteforce(self, pts, ref):
        """O(n^3) hypervolume computation for 3D."""
        n = len(pts)
        if n == 0:
            return 0.0

        # Sort by x
        pts = pts[np.argsort(pts[:, 0])]

        # For each point i, compute the volume of the box:
        # [pts[i].x, next_x) x [pts[i].y, y_max) x [pts[i].z, z_max)
        # where next_x is the x of the next point (or ref[0])
        # and y_max, z_max are the minimum of ref[1], ref[2] and the y, z of points
        # that dominate point i in y and z respectively.

        # Actually, the correct formula for the contribution of point i:
        # The box for point i is [pts[i].x, x_next) x [pts[i].y, y_bound) x [pts[i].z, z_bound)
        # where x_next = pts[i+1].x if i+1 < n else ref[0]
        # y_bound = min(ref[1], min of pts[j].y for all j where pts[j].y < pts[i].y and pts[j].z < pts[i].z)
        #          but this isn't right because we need the union.

        # Let me just use the inclusion-exclusion approach directly.
        # HV = sum of all boxes - sum of intersections + ...
        # This is O(2^n) which is too slow.

        # OK let me just implement the standard divide and conquer.
        # For 3D, we can do:
        # 1. Sort by x
        # 2. For each point i, the contribution is (x[i+1] - x[i]) * A_i
        #    where A_i is the 2D area of the union of rectangles [0, y_j) x [0, z_j)
        #    for all j >= i.

        # Wait, that's the key insight! Let me verify.
        # The HV is the volume of the union of boxes [x_i, ref[0]) x [y_i, ref[1]) x [z_i, ref[2])
        # for all i.
        # Sort by x. For x in [x_i, x_{i+1}), the active points are those with index >= i
        # (since they have x <= current x). The cross-section at this x is the union of
        # rectangles [y_j, ref[1]) x [z_j, ref[2]) for all j >= i.
        # So HV = sum_{i=0}^{n-1} (x_{i+1} - x_i) * Area(union of rects for j >= i)
        # where x_n = ref[0].

        # This is O(n^2) if we can compute the 2D area efficiently.
        # For 2D, the area of union of rectangles can be computed in O(n log n).

        x_coords = np.append(pts[:, 0], ref[0])

        total_hv = 0.0
        for i in range(n):
            # Get all points j >= i
            sub_pts = pts[i:]
            # Compute 2D area of union of rectangles [y_j, ref[1]) x [z_j, ref[2])
            area = self._hv_2d(sub_pts[:, 1:3], np.array([ref[1], ref[2]]))
            width = x_coords[i + 1] - x_coords[i]
            total_hv += width * area

        return float(total_hv)

    def _hv_2d(self, pts, ref):
        """2D hypervolume (area) computation."""
        n = len(pts)
        if n == 0:
            return 0.0

        # Filter points that are strictly better than ref
        mask = np.all(pts < ref, axis=1)
        pts = pts[mask]
        n = len(pts)
        if n == 0:
            return 0.0

        # Sort by y
        pts = pts[np.argsort(pts[:, 0])]

        # Sweep line: for each point, the contribution is (y_{i+1} - y_i) * z_min
        # where z_min is the minimum z among points with index >= i.

        y_coords = np.append(pts[:, 0], ref[0])
        # For each i, z_min_i = min(pts[j, 1] for j >= i)
        z_mins = np.minimum.accumulate(pts[:, 1][::-1])[::-1]
        # Pad with ref[1]
        z_mins = np.append(z_mins, ref[1])

        widths = y_coords[1:] - y_coords[:-1]
        # z_mins[i] is the min z for points with index >= i
        # The contribution for slab i is width_i * (ref[1] - z_mins[i])
        # Wait, the area is the integral of (ref[1] - z_min(x)) dx
        # where z_min(x) is the min z among active points at x.

        # Actually, for 2D HV:
        # Area = sum_{i=0}^{n-1} (y_{i+1} - y_i) * (ref[1] - z_min_i)
        # where z_min_i = min(pts[j, 1] for j >= i)

        areas = (ref[1] - z_mins[:-1]) * widths
        # Clip to non-negative
        areas = np.maximum(areas, 0)

        return float(np.sum(areas))

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        # MOEA/D parameters
        N = 30  # population size
        T = 5   # neighborhood size
        K = 10  # max number of updates per generation

        # Weight vectors: uniform distribution on the simplex
        # For 3 objectives, generate N points on the 2D simplex
        weights = []
        # Use a grid on the simplex
        n_grid = 5  # 5x5 grid gives ~15 points, we need 30
        # Let's use a different approach: random points on the simplex
        for _ in range(N):
            w = rng.random(n_obj)
            w /= w.sum()
            weights.append(w)
        weights = np.array(weights)

        # Initialize population
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N

        # Record all evaluated points
        all_evaluated = [F.copy()]

        # Neighborhood: for each weight, find T closest weights
        neighborhoods = []
        for i in range(N):
            dists = np.sum((weights - weights[i]) ** 2, axis=1)
            dists[i] = np.inf
            nearest = np.argsort(dists)[:T]
            neighborhoods.append(nearest)

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # For each solution, try to update it using a neighbor's parent
            updates = []
            for i in range(N):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Pick a random neighbor
                j = neighborhoods[i][rng.integers(T)]

                # Crossover
                if rng.random() < 0.9:
                    # SBX
                    beta = 1.0
                    eta_c = 20.0
                    child = np.zeros(dim)
                    for d in range(dim):
                        u = rng.random()
                        if u <= 1.0 / (beta + 1.0):
                            bc = (2 * u) ** (1.0 / (beta + 1.0))
                        elif u >= 1.0 - 1.0 / (beta + 1.0):
                            bc = (1.0 / (2 * (1 - u))) ** (1.0 / (beta + 1.0))
                        else:
                            bc = 1.0
                        child[d] = 0.5 * ((1 + bc) * X[i, d] + (1 - bc) * X[j, d])
                else:
                    child = X[i].copy()

                # Mutation
                eta_m = 20.0
                rate = 1.0 / dim
                for d in range(dim):
                    if rng.random() < rate:
                        u = rng.random()
                        if u < 0.5:
                            delta1 = (2 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                        else:
                            delta1 = 1.0 - (2 * (1 - u)) ** (1.0 / (eta_m + 1.0))
                        child[d] += delta1 * (hi[d] - lo[d])
                child = np.clip(child, lo, hi)

                # Evaluate
                fc = problem(child)
                used += 1

                # Update: accept if it improves the aggregated objective
                # or if it is non-dominated
                old_f = F[i]
                new_f = fc

                # Check if new_f dominates old_f
                if np.all(new_f <= old_f) and np.any(new_f < old_f):
                    X[i] = child
                    F[i] = new_f
                    updates.append(i)
                # Check if old_f dominates new_f - don't update
                elif np.all(old_f <= new_f) and np.any(old_f < new_f):
                    pass
                # Otherwise, use aggregated objective
                else:
                    old_agg = np.sum(weights[i] * old_f)
                    new_agg = np.sum(weights[i] * new_f)
                    if new_agg < old_agg:
                        X[i] = child
                        F[i] = new_f
                        updates.append(i)

                # Record
                all_evaluated.append(F.copy())

        return X
# EVOLVE-BLOCK-END
