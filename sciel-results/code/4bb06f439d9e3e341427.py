import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self._start_time = None

    def __call__(self, problem, bounds):
        self._start_time = time.time()
        deadline = self._start_time + 290  # 290s safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        ref = np.array([1.1, 1.3, 1.1])
        
        rng = np.random.default_rng(0)
        
        # Archive: list of (x, f) tuples
        archive_x = []
        archive_f = []
        used = 0
        
        def _is_dominated(f_new, archive_f):
            """Check if f_new is dominated by any point in archive_f."""
            for f in archive_f:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False
        
        def _dominates(f_new, f_other):
            """Check if f_new strictly dominates f_other."""
            return np.all(f_new <= f_other) and np.any(f_new < f_other)
        
        def _update_archive(x, f):
            """Add point to archive if non-dominated, remove dominated points."""
            nonlocal archive_x, archive_f
            # Check if dominated
            if _is_dominated(f, archive_f):
                return
            # Remove dominated points
            new_x = []
            new_f = []
            for i in range(len(archive_x)):
                if not _dominates(f, archive_f[i]):
                    new_x.append(archive_x[i])
                    new_f.append(archive_f[i])
            # Add new point
            new_x.append(x)
            new_f.append(f)
            archive_x = new_x
            archive_f = new_f
        
        def _hypervolume_3d(points, ref):
            """Compute exact 3D hypervolume for non-dominated points."""
            if len(points) == 0:
                return 0.0
            # Filter points that are within reference
            pts = [p for p in points if np.all(p < ref)]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            pts = sorted(pts, key=lambda p: p[0])
            # For each point, compute its contribution
            hv = 0.0
            for i, p in enumerate(pts):
                # The region contributed by point i is the box from p to ref,
                # but we need to subtract regions already covered by points with smaller obj[0]
                # Actually, for non-dominated points sorted by obj[0] ascending,
                # the contribution of point i is:
                # (ref[0] - p[0]) * HV_2D of the "shadow" in obj[1], obj[2]
                # But this is complex. Let's use a different approach.
                # 
                # Alternative: use the standard 3D HV algorithm.
                # Sort by obj[0] descending.
                # For each point, the contribution is the volume of the box
                # [p[0], ref[0]] x [p[1], ref[1]] x [p[2], ref[2]]
                # minus the parts already covered by points with smaller obj[0].
                
                pass
            # Let's use a simpler but exact approach for small n.
            # Brute force: for each non-dominated point, compute the volume of
            # the box [p, ref] that is not covered by any other point.
            # This is O(n^3) which is fine for n <= 200.
            
            # Actually, let's use the standard recursive approach.
            # Sort by first objective ascending.
            # The hypervolume is the sum over all points of the "unique" volume.
            
            # For 3D, a clean approach:
            # Sort points by f0 ascending.
            # For point i, the unique contribution is the volume of the box
            # [f_i, ref] intersected with the region where no other point dominates.
            # Since points are non-dominated, for point i, the region where it's
            # the "best" in f0 is [f_i[0], f_{i+1}[0]] (or [f_i[0], ref[0]] for last).
            # Within that f0 slice, we need the 2D hypervolume of the points'
            # (f1, f2) values.
            
            if len(pts) == 1:
                return np.prod(ref - pts[0])
            
            # Sort by f0 ascending
            pts_sorted = sorted(pts, key=lambda p: p[0])
            hv = 0.0
            for i in range(len(pts_sorted)):
                p = pts_sorted[i]
                # f0 range: from p[0] to next point's f0 (or ref[0])
                if i < len(pts_sorted) - 1:
                    f0_next = pts_sorted[i+1][0]
                else:
                    f0_next = ref[0]
                
                if f0_next <= p[0]:
                    continue
                
                # In the f0 slice [p[0], f0_next], the "active" points are
                # those with f0 <= f0_next. Since we sorted ascending, that's
                # points 0..i. But for the 2D HV, we only care about the
                # (f1, f2) of points that are non-dominated in 2D.
                # 
                # Actually, the contribution of point i in this slice is:
                # (f0_next - p[0]) * HV_2D of the set of (f1, f2) for points
                # j where f_j[0] <= f0_next, but we need to be careful.
                # 
                # Let me use a different, cleaner algorithm.
                
                pass
            
            # Let me just use a direct O(n^3) approach for correctness.
            # For each point p, the unique volume is the volume of the box [p, ref]
            # minus the volume of the intersection with boxes of other points.
            # This is complex. Let's use a library-like approach.
            
            # Actually, for 3D with small n, we can use the following:
            # The hypervolume is the integral over the box [0,ref] of the
            # indicator that the point is not dominated.
            # 
            # Let me just implement a simple but correct 3D HV.
            # Sort by f0 ascending. For each point i, the contribution is:
            # (f0_{i+1} - f0_i) * HV_2D(points 0..i in f1,f2 space, ref=[ref1, ref2])
            # where f0_{n} = ref[0].
            
            pts_sorted = sorted(pts, key=lambda p: p[0])
            n_pts = len(pts_sorted)
            hv = 0.0
            for i in range(n_pts):
                p = pts_sorted[i]
                f0_hi = pts_sorted[i+1][0] if i + 1 < n_pts else ref[0]
                if f0_hi <= p[0]:
                    continue
                # 2D HV of points 0..i in (f1, f2) space
                # The points are (p_j[1], p_j[2]) for j=0..i
                # We need the 2D hypervolume with reference (ref[1], ref[2])
                hv2 = self._hv_2d([pts_sorted[j][1:2+1] for j in range(i+1)], ref[1:3])
                hv += (f0_hi - p[0]) * hv2
            
            return hv
        
        def _hv_2d(points, ref):
            """Compute 2D hypervolume for non-dominated points."""
            if len(points) == 0:
                return 0.0
            pts = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
            if len(pts) == 0:
                return 0.0
            # Sort by f0 ascending
            pts = sorted(pts, key=lambda p: p[0])
            hv = 0.0
            for i, p in enumerate(pts):
                f0_hi = pts[i+1][0] if i + 1 < len(pts) else ref[0]
                if f0_hi <= p[0]:
                    continue
                # The height in f1 is ref[1] - p[1]
                # But we need to account for other points.
                # In 2D, the contribution of point i is (f0_hi - p[0]) * (ref[1] - p[1])
                # only if no other point with smaller f0 has a smaller f1.
                # Actually, for non-dominated points sorted by f0 ascending,
                # f1 is strictly decreasing. So the contribution is:
                # (f0_hi - p[0]) * (ref[1] - p[1])
                hv += (f0_hi - p[0]) * (ref[1] - p[1])
            return hv
        
        def _marginal_hv(x, f):
            """Compute marginal hypervolume contribution of point (x,f) to current archive."""
            if _is_dominated(f, archive_f):
                return 0.0
            # Hypervolume with the new point
            new_archive_f = list(archive_f)
            new_archive_f.append(f)
            # Remove dominated points from new_archive_f
            filtered = []
            for i in range(len(new_archive_f)):
                dominated = False
                for j in range(len(new_archive_f)):
                    if i == j:
                        continue
                    if _dominates(new_archive_f[j], new_archive_f[i]):
                        dominated = True
                        break
                if not dominated:
                    filtered.append(new_archive_f[i])
            hv_with = _hypervolume_3d(filtered, ref)
            hv_without = _hypervolume_3d(archive_f, ref)
            return hv_with - hv_without
        
        # Phase 1: Initial random sampling
        n_init = min(20, budget)
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(d)
            f = problem(x)
            used += 1
            _update_archive(x, f)
        
        # Phase 2: Greedy hypervolume maximization
        # Generate a dense candidate lattice
        n_per_dim = 6  # 6^4 = 1296 candidates
        candidates = []
        for x0 in np.linspace(lo[0], hi[0], n_per_dim):
            for x1 in np.linspace(lo[1], hi[1], n_per_dim):
                for x2 in np.linspace(lo[2], hi[2], n_per_dim):
                    for x3 in np.linspace(lo[3], hi[3], n_per_dim):
                        candidates.append(np.array([x0, x1, x2, x3]))
        
        # Also add some random candidates
        for i in range(100):
            candidates.append(lo + (hi - lo) * rng.random(d))
        
        # Evaluate candidates greedily
        while used < budget:
            if time.time() > deadline:
                break
            
            best_candidate = None
            best_marginal = -np.inf
            best_f = None
            
            for c in candidates:
                if time.time() > deadline:
                    break
                f = problem(c)
                used += 1
                marginal = _marginal_hv(c, f)
                if marginal > best_marginal:
                    best_marginal = marginal
                    best_candidate = c
                    best_f = f
            
            if best_candidate is not None:
                _update_archive(best_candidate, best_f)
                # Remove the chosen candidate from the list
                candidates.remove(best_candidate)
            
            if not candidates:
                break
        
        # Phase 3: Local refinement around archive points
        while used < budget:
            if time.time() > deadline:
                break
            
            # Pick a random archive point and perturb it
            if not archive_x:
                break
            idx = rng.integers(len(archive_x))
            x_base = archive_x[idx]
            
            # Gaussian perturbation
            sigma = 0.05
            x = x_base + sigma * rng.standard_normal(d)
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            used += 1
            _update_archive(x, f)
        
        # Return all evaluated points (the archive)
        return np.array(archive_x) if archive_x else np.empty((0, d))
