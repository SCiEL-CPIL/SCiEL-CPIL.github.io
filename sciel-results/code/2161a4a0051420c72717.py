"""Multi-objective optimizer on RE21 using archive-based hypervolume-guided search."""
import time
import numpy as np


class Optimizer:
    """Archive-based multi-objective optimizer that maximizes hypervolume contribution."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        ref = np.array([3000.0, 0.06])

        # Archive: list of (x, f) where f is strictly better than ref in all objectives
        archive = []  # list of [x_array, f_array]
        used = 0

        def is_dominated(f_new, archive):
            """Check if f_new is dominated by any point in archive."""
            for _, f_old in archive:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return True
            return False

        def hv_contribution(f_new, archive):
            """Compute the hypervolume contribution of f_new given the archive.
            In 2D, the contribution is the area of the region that f_new covers
            that is not covered by any other point in the archive.
            """
            # For 2D minimization with reference point ref:
            # A point p contributes HV = (ref[0]-p[0]) * (ref[1]-p[1]) if it's on the Pareto front
            # But we need the marginal contribution.
            
            # In 2D, if we sort points by f1, the HV of the set is the sum of rectangles.
            # The marginal contribution of a new point is:
            # width * height where width is determined by neighboring points on the Pareto front.
            
            # Simpler approach: compute HV before and after adding the point.
            # But that's expensive. For 2D, we can compute it directly.
            
            # Let's compute the HV of the archive + new point minus HV of archive alone.
            # For efficiency in 2D, we can sort by first objective.
            
            # Actually, let's just use a simple heuristic: the "area" the point covers
            # that isn't covered by others. In 2D, if we have the Pareto front sorted by f1,
            # each point i covers a rectangle from f1_i to f1_{i+1} in x, and from f2_i to ref[1] in y.
            
            # Let me compute it properly:
            # Collect all points (archive + new) that are non-dominated and better than ref
            all_points = [f for _, f in archive]
            all_points.append(f_new)
            
            # Filter: must be strictly better than ref in all objectives
            valid = [p for p in all_points if np.all(p < ref)]
            if len(valid) == 0:
                return 0.0
            
            # Find non-dominated points among valid
            nondom = []
            for i, p in enumerate(valid):
                dominated = False
                for j, q in enumerate(valid):
                    if i == j:
                        continue
                    if np.all(q <= p) and np.any(q < p):
                        dominated = True
                        break
                if not dominated:
                    nondom.append(p)
            
            if len(nondom) == 0:
                return 0.0
            
            # Sort by first objective
            nondom.sort(key=lambda p: p[0])
            
            # Compute HV for this set
            def compute_hv_2d(points, ref):
                """Compute HV for 2D minimization with reference point."""
                if not points:
                    return 0.0
                # Sort by first objective
                pts = sorted(points, key=lambda p: p[0])
                # Remove duplicates and dominated
                filtered = []
                for p in pts:
                    if not filtered or p[1] < filtered[-1][1]:
                        filtered.append(p)
                    else:
                        # p is dominated by filtered[-1] in the sorted order
                        # Actually not quite, but since sorted by p[0], if p[1] >= filtered[-1][1], 
                        # then filtered[-1] dominates p
                        pass
                # Compute HV
                hv = 0.0
                for i, p in enumerate(filtered):
                    x_left = p[0]
                    x_right = ref[0] if i == len(filtered) - 1 else filtered[i + 1][0]
                    y_bottom = p[1]
                    y_top = ref[1]
                    if x_right > x_left and y_top > y_bottom:
                        hv += (x_right - x_left) * (y_top - y_bottom)
                return hv
            
            # HV with new point
            hv_with = compute_hv_2d(nondom, ref)
            
            # HV without new point
            archive_valid = [f for _, f in archive if np.all(f < ref)]
            if archive_valid:
                archive_nondom = []
                for i, p in enumerate(archive_valid):
                    dominated = False
                    for j, q in enumerate(archive_valid):
                        if i == j:
                            continue
                        if np.all(q <= p) and np.any(q < p):
                            dominated = True
                            break
                    if not dominated:
                        archive_nondom.append(p)
                hv_without = compute_hv_2d(archive_nondom, ref)
            else:
                hv_without = 0.0
            
            return hv_with - hv_without

        def add_to_archive(x, f):
            """Add point to archive if non-dominated and better than ref. Remove dominated points."""
            nonlocal used
            if not np.all(f < ref):
                return
            # Check if dominated by existing archive points
            if is_dominated(f, archive):
                return
            # Remove points dominated by new point
            new_archive = []
            for ax, af in archive:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append([ax, af])
            new_archive.append([x, f])
            archive = new_archive

        # Initialize with random points
        n_init = min(20, self.budget // 5)
        for i in range(n_init):
            if used >= self.budget or time.time() > deadline:
                break
            x = lo + span * np.random.rand(self.dim)
            f = np.array(problem(x), dtype=np.float64)
            used += 1
            add_to_archive(x, f)

        # Main loop: sample from archive, perturb, evaluate
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Choose a parent from the archive
            if archive:
                # Weight by hypervolume contribution (approximate)
                # For simplicity, pick randomly from non-dominated archive
                parent_idx = np.random.randint(len(archive))
                parent_x = archive[parent_idx][0].copy()
            else:
                parent_x = lo + span * np.random.rand(self.dim)
            
            # Adaptive mutation: scale based on archive spread
            if len(archive) > 1:
                f_arr = np.array([f for _, f in archive])
                spread = np.ptp(f_arr, axis=0)
                # Use a moderate mutation rate
                mut_scale = 0.15
            else:
                mut_scale = 0.3
            
            # Generate child
            child_x = parent_x + mut_scale * span * np.random.randn(self.dim)
            child_x = np.clip(child_x, lo, hi)
            
            f_child = np.array(problem(child_x), dtype=np.float64)
            used += 1
            
            # Add to archive if it's useful
            if np.all(f_child < ref):
                # Check if dominated
                dominated = False
                for _, af in archive:
                    if np.all(af <= f_child) and np.any(af < f_child):
                        dominated = True
                        break
                if not dominated:
                    # Remove dominated points
                    new_archive = []
                    for ax, af in archive:
                        if not (np.all(f_child <= af) and np.any(f_child < af)):
                            new_archive.append([ax, af])
                    new_archive.append([child_x, f_child])
                    archive = new_archive

        return archive

    def _hv_contribution(self, f_new, archive, ref):
        """Compute marginal hypervolume contribution of f_new in 2D."""
        # This is a helper that could be optimized
        pass
