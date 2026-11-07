# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using Sobol quasi-random initialization
    followed by hypervolume-contribution-based local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        rng = np.random.default_rng(42)

        # Reference point for scoring
        ref = np.array([1705.0, 12.0, 0.3])

        # --- Phase 1: Sobol quasi-random initialization (150 evaluations) ---
        # Generate Sobol sequence for uniform coverage
        from scipy.stats import qmc

        n_init = 150
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        # We need 2**k points; find smallest k such that 2**k >= n_init
        k = int(np.ceil(np.log2(n_init)))
        n_points = 2 ** k
        sobol_samples = sampler.random(n_points)
        # Map to bounds
        sobol_samples = lo + sobol_samples * (hi - lo)

        archive_X = []
        archive_F = []
        used = 0

        def dominates(f1, f2):
            """Check if f1 dominates f2 (all objectives <= and at least one <)."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F."""
            for af in archive_F:
                if dominates(af, f):
                    return True
            return False

        def update_archive(archive_X, archive_F, x, f):
            """Update archive with new point if non-dominated. Returns updated archives."""
            if is_dominated(f, archive_F):
                return archive_X, archive_F, False

            # Remove dominated points from archive
            new_X = [x]
            new_F = [f]
            for i in range(len(archive_F)):
                if not dominates(f, archive_F[i]):
                    new_X.append(archive_X[i])
                    new_F.append(archive_F[i])

            return new_X, new_F, True

        # Evaluate Sobol points
        for i in range(min(n_init, n_points)):
            if used >= budget or time.time() > deadline - 30:
                break
            x = sobol_samples[i]
            f = problem(x)
            used += 1
            archive_X, archive_F, _ = update_archive(archive_X, archive_F, x, f)

        # Convert to numpy arrays
        if len(archive_X) > 0:
            archive_X = np.array(archive_X)
            archive_F = np.array(archive_F)
        else:
            archive_X = np.empty((0, dim))
            archive_F = np.empty((0, n_obj))

        # --- Phase 2: Hypervolume-contribution-based local search (remaining budget) ---
        def compute_hypervolume_contribution(archive_F, ref):
            """Compute hypervolume contribution of each point in the archive.
            Returns array of contributions (hypervolume that would be lost if point removed)."""
            n = len(archive_F)
            if n == 0:
                return np.array([])

            def hypervolume_3d(points, ref):
                """Compute exact hypervolume for 3D points relative to reference point."""
                if len(points) == 0:
                    return 0.0
                # Filter points strictly better than ref in all objectives
                mask = np.all(points < ref, axis=1)
                pts = points[mask]
                if len(pts) == 0:
                    return 0.0

                # Sort by first objective
                idx = np.argsort(pts[:, 0])
                pts = pts[idx]

                # For 3D, we can compute HV by decomposing into slabs
                # Sort by x, then for each point, compute the area in y-z plane
                # that is not covered by subsequent points
                hv = 0.0
                prev_x = pts[0, 0]
                # We need to compute the union of boxes
                # For 3D, use a sweep line approach

                # Actually, let's use a simpler approach for small n
                # Sort by x ascending
                # For each point i, the contribution is (ref[0] - pts[i,0]) * area_in_yz
                # where area_in_yz is the area of the region in y-z that is only
                # covered by point i and not by any point j with pts[j,0] <= pts[i,0]
                # This is complex. Let's use a different approach.

                # For 3D with small n, we can use inclusion-exclusion or
                # just compute the hypervolume directly by sorting and sweeping.

                # Sort by x
                order = np.argsort(pts[:, 0])
                pts = pts[order]

                # Sweep from left to right (increasing x)
                # Maintain a set of "active" y-z rectangles
                # This is getting complex. Let's use a simpler numerical approach
                # or just compute HV by decomposing into non-overlapping boxes.

                # For 3D, sort by x, then for each point, the box it contributes
                # is from x to next x (or ref[0]), y from pts[i,1] to max of
                # subsequent points' y that are < ref[1], etc.

                # Actually, let's just use a standard 3D HV algorithm:
                # Sort by first objective
                # Use a recursive divide and conquer or sweep line

                # For simplicity and correctness with small n, let's compute
                # the hypervolume by enumerating all possible boxes
                # This is O(n^3) which is fine for n < 50

                # Alternative: use the fact that in 3D, we can sort by x,
                # then for each "layer" in x, compute the 2D hypervolume in y-z

                # Sort by x
                pts_sorted = pts[np.argsort(pts[:, 0])]

                # Compute hypervolume by sweeping
                # For each point in order of increasing x, compute the area in y-z
                # that this point "owns" (not covered by any point with smaller x)

                # This is equivalent to computing the union of boxes
                # Let's use a simple approach: for each point, compute the box
                # it would define if it were the only point, then subtract overlaps.

                # Actually, the cleanest way for 3D:
                # Sort by x. For each i, the contribution in x is (x_{i+1} - x_i)
                # where x_{n} = ref[0]. For that slab, compute the 2D hypervolume
                # of points with index <= i in the y-z plane.

                n_pts = len(pts_sorted)
                hv = 0.0
                for i in range(n_pts):
                    x_start = pts_sorted[i, 0]
                    x_end = pts_sorted[i + 1, 0] if i + 1 < n_pts else ref[0]
                    dx = x_end - x_start
                    if dx <= 0:
                        continue

                    # In this x-slab, the active points are 0..i
                    # Compute 2D hypervolume of points[0:i+1] in y-z plane
                    # relative to (ref[1], ref[2])
                    active = pts_sorted[:i + 1, 1:3]
                    area = _hypervolume_2d(active, ref[1:3])
                    hv += dx * area

                return hv

            def _hypervolume_2d(points, ref2d):
                """Compute 2D hypervolume of points relative to ref2d."""
                if len(points) == 0:
                    return 0.0
                # Sort by first coordinate
                idx = np.argsort(points[:, 0])
                pts = points[idx]

                # Sweep: for each point, compute the width in y that it owns
                hv = 0.0
                prev_y = ref2d[1]
                for i in range(len(pts)):
                    # The y-range owned by point i is from pts[i,1] to prev_y
                    # if pts[i,1] < prev_y
                    y_start = pts[i, 1]
                    y_end = prev_y
                    if y_end > y_start:
                        width = ref2d[0] - pts[i, 0]
                        hv += width * (y_end - y_start)
                    if pts[i, 1] < prev_y:
                        prev_y = pts[i, 1]
                return hv

            contributions = np.zeros(n)
            for i in range(n):
                # Hypervolume with point i
                hv_with = hypervolume_3d(archive_F, ref)
                # Hypervolume without point i
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                hv_without = hypervolume_3d(archive_F[mask], ref)
                contributions[i] = hv_with - hv_without

            return contributions

        # Local search refinement
        remaining = budget - used
        if remaining > 0 and len(archive_X) > 0:
            # We'll do local search on each archive point
            # For each point, try perturbations and keep if it improves HV contribution
            n_archive = len(archive_X)

            # Adaptive step size
            base_step = 0.1 * (hi - lo)

            for _ in range(remaining):
                if used >= budget or time.time() > deadline - 10:
                    break

                # Pick a random archive point to refine
                idx = rng.integers(n_archive)
                x = archive_X[idx].copy()
                f = archive_F[idx].copy()

                # Generate perturbation
                step = base_step * np.exp(-0.01 * used)  # decay step size
                perturbation = step * rng.standard_normal(dim)
                x_new = np.clip(x + perturbation, lo, hi)
                f_new = problem(x_new)
                used += 1

                # Check if new point is non-dominated and improves archive
                if not is_dominated(f_new, archive_F):
                    # Update archive
                    archive_X, archive_F, added = update_archive(archive_X, archive_F, x_new, f_new)
                    if added:
                        n_archive = len(archive_X)

        # Return the archive
        return archive_X

# EVOLVE-BLOCK-END
