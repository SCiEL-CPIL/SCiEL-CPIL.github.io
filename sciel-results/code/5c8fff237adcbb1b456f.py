"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-Driven Multi-Objective Evolutionary Strategy."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, a, b):
        """Return True if a dominates b (all <=, at least one <)."""
        return np.all(a <= b) and np.any(a < b)

    def _is_nondominated(self, point, archive_f):
        """Check if point is non-dominated w.r.t. list of objective vectors."""
        for a in archive_f:
            if self._dominates(a, point):
                return False
        return True

    def _hv_contribution_3d(self, f, archive_f, ref):
        """
        Compute hypervolume contribution of point f to the set archive_f.
        Uses the standard 3D HV contribution formula.
        Returns 0 if f is dominated by any point in archive_f.
        """
        # Check if f is dominated
        for a in archive_f:
            if self._dominates(a, f):
                return 0.0
        
        # Compute contribution: volume of region dominated by f but not by any other point
        # For 3D, sort archive points by f[0], then compute contributions
        points = [(f[0], f[1], f[2])] + [(a[0], a[1], a[2]) for a in archive_f]
        points.sort(key=lambda p: p[0])
        
        contrib = 0.0
        for i, (x1, x2, x3) in enumerate(points):
            if x1 >= ref[0] or x2 >= ref[1] or x3 >= ref[2]:
                continue
            # For each point, compute the volume it uniquely contributes
            # This is the volume of the box [x1, ref[0]] x [x2, ref[1]] x [x3, ref[2]]
            # minus overlaps with other points that dominate this region
            
            # Simpler approach: compute the union of boxes and find f's unique part
            # For efficiency, use the incremental method
            pass
        
        # Use a simpler but correct approach: compute HV of archive_f and HV of archive_f + {f}
        # This is O(n^2) but n is small
        hv_before = self._compute_hv(archive_f, ref)
        hv_after = self._compute_hv(archive_f + [f], ref)
        return hv_after - hv_before

    def _compute_hv(self, points, ref):
        """
        Compute exact hypervolume for 3D set of points.
        points: list of (f1, f2, f3) tuples
        ref: reference point [r1, r2, r3]
        Returns HV value.
        """
        if not points:
            return 0.0
        
        # Filter points that are within the reference box
        valid = [(p[0], p[1], p[2]) for p in points 
                 if p[0] < ref[0] and p[1] < ref[1] and p[2] < ref[2]]
        if not valid:
            return 0.0
        
        # Sort by first objective
        valid.sort(key=lambda p: p[0])
        
        # Use the standard 3D HV algorithm
        # For each point, compute the volume it contributes
        # This is done by sweeping through points sorted by f1
        hv = 0.0
        n = len(valid)
        
        # For each point i, compute the volume of the box it uniquely dominates
        # The unique contribution of point i is the volume of:
        # [valid[i][0], valid[i+1][0] or ref[0]] x [valid[i][1], ref[1]] x [valid[i][2], ref[2]]
        # minus parts already covered by previous points
        
        # More robust: use the recursive decomposition
        # Or use the simple O(n^2) method for small n
        
        # Simple method: for each point, compute the volume of the box it spans
        # and subtract overlaps
        # For 3D, we can use the inclusion-exclusion or just compute directly
        
        # Direct computation: for each point, the box it defines is
        # [p1, r1] x [p2, r2] x [p3, r3]
        # The HV is the union of all these boxes
        
        # Use a sweep line approach for efficiency
        # Sort by f1
        # For each point, the contribution is the volume of the 2D region in (f2, f3)
        # that is not covered by any point with smaller f1
        
        # For small n, use brute force with careful bookkeeping
        # We'll use the standard algorithm: sort by f1, then for each point,
        # compute the 2D HV in the (f2, f3) plane for the remaining points
        
        # Actually, let's use a simpler but correct O(n^2) method
        # For each pair of points, compute their contribution
        
        # Best approach for small n: use the formula
        # HV = sum over all subsets... no, that's exponential
        
        # Let's use the sweep line properly
        # Sort by f1 ascending
        # Maintain a data structure of (f2, f3) points
        # For each point in order, compute the 2D HV of the region
        # [f2, r2] x [f3, r3] that is not covered by previous points
        
        # For n < 50, we can use a simpler method
        # Compute the 2D HV for each "slice"
        
        # I'll use a direct O(n^2) method that works for small archives
        # For each point i, compute the volume of the box it uniquely contributes
        # by checking against all other points
        
        # Actually, the cleanest correct method for 3D with small n:
        # Use the recursive formula or just compute the union volume
        
        # Let me use a well-known O(n^2 log n) or O(n^2) method
        # For 3D, we can do:
        # 1. Sort by f1
        # 2. For each point i from left to right, compute the 2D HV in (f2, f3)
        #    of the points j where j > i (in sorted order) and f2_j < r2, f3_j < r3
        #    but this isn't quite right either
        
        # Let me just use a simple but correct O(n^2) method:
        # For each point, compute the volume of the box [p1, r1] x [p2, r2] x [p3, r3]
        # and use inclusion-exclusion... but that's complex
        
        # For practical purposes with n < 50, let's use a Monte Carlo-free exact method
        # by computing the 2D HV for each slice
        
        # Here's a correct O(n^2) method for 3D:
        # Sort points by f1
        # For each point i, the contribution is:
        # (r1 - f1_i) * (2D HV of points j with f1_j > f1_i in the (f2, f3) plane,
        #   clipped to [f2_i, r2] x [f3_i, r3])
        # Wait, that's not right either
        
        # Let me use a different approach: the "layer cake" method
        # Or just use scipy's nD HV if available... but we should be self-contained
        
        # For now, use a simple but potentially slow O(n^2) method:
        # Compute the volume of the union of boxes
        # Each box is [p1, r1] x [p2, r2] x [p3, r3]
        # Use the inclusion-exclusion principle... but for n>10 it's impractical
        
        # Practical solution: use a grid-based approximation? No, we need exact.
        
        # Let me implement the standard 3D HV algorithm properly:
        # 1. Sort by f1
        # 2. Sweep from left to right
        # 3. At each step, compute the 2D HV of the remaining points in (f2, f3)
        #    for the region [current_f2, r2] x [current_f3, r3]
        
        # Actually, the correct algorithm is:
        # HV = sum over i of (f1_{i+1} - f1_i) * HV_2D(points with f1 >= f1_i, in (f2,f3) plane)
        # where f1_{n+1} = r1
        # and HV_2D is the 2D hypervolume
        
        # Let me implement this properly
        n_valid = len(valid)
        if n_valid == 0:
            return 0.0
        
        # Sort by f1
        valid.sort(key=lambda p: p[0])
        
        hv = 0.0
        for i in range(n_valid):
            # The slice from valid[i][0] to valid[i+1][0] (or ref[0])
            f1_next = valid[i+1][0] if i + 1 < n_valid else ref[0]
            f1_width = f1_next - valid[i][0]
            if f1_width <= 0:
                continue
            
            # Compute 2D HV of points with index >= i in the (f2, f3) plane
            # clipped to [valid[i][1], ref[1]] x [valid[i][2], ref[2]]
            # Actually, we need the 2D HV of all points j >= i
            # in the region [min_f2, ref[1]] x [min_f3, ref[2]]
            # where min_f2 = min of f2_j for j >= i, etc.
            # But the standard formula uses the 2D HV of the points in the (f2,f3) plane
            # with reference point (ref[1], ref[2])
            
            # For the slice at f1 = valid[i][0], the 2D HV is computed from
            # all points j where j >= i (in sorted order)
            sub_points = [valid[j][1:3] for j in range(i, n_valid)
                         if valid[j][1] < ref[1] and valid[j][2] < ref[2]]
            
            hv_2d = self._compute_hv_2d(sub_points, ref[1], ref[2])
            hv += f1_width * hv_2d
        
        return hv

    def _compute_hv_2d(self, points, ref2, ref3):
        """Compute 2D hypervolume for points in (f2, f3) plane."""
        if not points:
            return 0.0
        
        # Filter valid points
        valid = [(p[0], p[1]) for p in points if p[0] < ref2 and p[1] < ref3]
        if not valid:
            return 0.0
        
        # Sort by f2
        valid.sort(key=lambda p: p[0])
        
        hv = 0.0
        n = len(valid)
        for i in range(n):
            f2_next = valid[i+1][0] if i + 1 < n else ref2
            f2_width = f2_next - valid[i][0]
            if f2_width <= 0:
                continue
            f3_height = ref3 - valid[i][1]
            if f3_height > 0:
                hv += f2_width * f3_height
        
        return hv

    def _add_to_archive(self, x, f, archive):
        """
        Add (x, f) to archive if f is non-dominated.
        Archive is a list of (x, f) tuples.
        Returns updated archive.
        """
        archive_f = [a[1] for a in archive]
        
        if not self._is_nondominated(f, archive_f):
            return archive
        
        # Remove dominated points
        new_archive = []
        for x_a, f_a in archive:
            if not self._dominates(f, f_a):
                new_archive.append((x_a, f_a))
        
        new_archive.append((x, f))
        return new_archive

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin (330s limit)
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        ref = np.array([3.0, 5.0, 7.0])

        total_budget = self.budget
        used = 0

        # Archive stores (x, f) tuples
        archive = []

        def evaluate(x):
            nonlocal used
            if used >= total_budget:
                return None
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            used += 1
            return f

        # Phase 1: Random Initialization
        n_init = min(30, total_budget // 6)
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = lo + range_ * rng.random(self.dim)
            f = evaluate(x)
            if f is None:
                break
            archive = self._add_to_archive(x, f, archive)
        
        if not archive:
            x = (lo + hi) / 2
            f = evaluate(x)
            if f is not None:
                archive.append((x, f))

        # Phase 2: Hypervolume-Driven Evolutionary Loop
        while used < total_budget and time.time() < deadline:
            # Select parent based on hypervolume contribution
            if len(archive) > 1:
                # Compute HV contribution for each archive point
                hv_contribs = []
                archive_f = [a[1] for a in archive]
                for i, (x_a, f_a) in enumerate(archive):
                    others = [archive_f[j] for j in range(len(archive_f)) if j != i]
                    contrib = self._hv_contribution_3d(f_a, others, ref)
                    hv_contribs.append(contrib)
                
                # Select parent with probability proportional to HV contribution
                total_contrib = sum(hv_contribs)
                if total_contrib > 0:
                    probs = np.array(hv_contribs) / total_contrib
                    idx = rng.choice(len(archive), p=probs)
                else:
                    idx = rng.integers(len(archive))
            else:
                idx = 0
            
            x_parent = archive[idx][0]
            
            # Adaptive mutation with Cauchy tails
            progress = used / total_budget
            base_scale = 0.15 * (1.0 - progress) + 0.02
            # Use Cauchy distribution for heavier tails
            noise = rng.standard_t(df=3, size=self.dim)
            noise = np.clip(noise, -3, 3)  # Clip extreme values
            x_child = x_parent + base_scale * range_ * noise
            x_child = np.clip(x_child, lo, hi)
            
            f_child = evaluate(x_child)
            if f_child is None:
                break
            
            archive = self._add_to_archive(x_child, f_child, archive)
            
            # Local search on points with high HV contribution
            if used < total_budget - 10:
                # Find the point with highest HV contribution
                archive_f = [a[1] for a in archive]
                best_idx = 0
                best_contrib = -1
                for i, (x_a, f_a) in enumerate(archive):
                    others = [archive_f[j] for j in range(len(archive_f)) if j != i]
                    contrib = self._hv_contribution_3d(f_a, others, ref)
                    if contrib > best_contrib:
                        best_contrib = contrib
                        best_idx = i
                
                x_cur = archive[best_idx][0].copy()
                f_cur = archive[best_idx][1].copy()
                
                # Local search with decreasing step size
                for ls_iter in range(4):
                    if used >= total_budget or time.time() > deadline:
                        break
                    ls_scale = 0.03 * (1.0 - ls_iter * 0.2)
                    x_trial = x_cur + ls_scale * range_ * rng.standard_normal(self.dim)
                    x_trial = np.clip(x_trial, lo, hi)
                    f_trial = evaluate(x_trial)
                    if f_trial is None:
                        break
                    if self._is_nondominated(f_trial, [a[1] for a in archive]):
                        archive = self._add_to_archive(x_trial, f_trial, archive)
                        x_cur = x_trial
                        f_cur = f_trial

        # Return objective vectors
        return [a[1] for a in archive]
# EVOLVE-BLOCK-END
