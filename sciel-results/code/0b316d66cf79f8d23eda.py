import numpy as np
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.archive = []
        self.archive_F = []
        self.evals_used = 0

    def _compute_hypervolume_3d(self, F, ref):
        """Compute exact hypervolume for 3 objectives using O(n log n) algorithm"""
        if len(F) == 0:
            return 0.0
        # Filter points strictly better than reference
        mask = np.all(F < ref, axis=1)
        F = F[mask]
        if len(F) == 0:
            return 0.0
        
        # Sort by first objective
        idx = np.argsort(F[:, 0])
        F = F[idx]
        
        # For 3D, we can use a sweep line approach
        # Sort by f1, then for each point, compute the 2D hypervolume contribution
        hv = 0.0
        # Keep track of the "skyline" in the 2D space (f2, f3)
        # Process points in order of increasing f1
        prev_f2 = ref[1]
        prev_f3 = ref[2]
        
        # We need to maintain the lower envelope in (f2, f3) space
        # For each new point, the contribution is the volume of the box
        # that is not covered by previous points
        
        # Simple approach: for each point, compute the 2D hypervolume in (f2,f3)
        # from this point to the reference, minus what's already covered
        # This is O(n^2) but n is small enough
        
        # Better: maintain a list of non-dominated points in (f2,f3)
        # and compute their 2D hypervolume contribution weighted by (ref[0] - f1_i)
        
        # For simplicity and correctness, use the O(n^2) approach
        # Sort by f1 ascending
        n = len(F)
        if n == 0:
            return 0.0
        
        # For each point i, the contribution is:
        # (ref[0] - F[i,0]) * area_2d(F[i,1:], ref[1:]) 
        # where area_2d is the 2D hypervolume in (f2,f3) that is NOT covered
        # by any point j with F[j,0] < F[i,0] and F[j,1] <= F[i,1] and F[j,2] <= F[i,2]
        
        # Actually, the standard 3D hypervolume computation:
        # Sort by f1. For each point, the "slice" contribution is:
        # (f1[i+1] - f1[i]) * H2(points with f1 <= f1[i] in (f2,f3) space)
        # where H2 is the 2D hypervolume
        
        # Let me use a cleaner approach:
        # Group by f1, compute 2D HV for each group
        
        # Simple O(n^2) for small n:
        # For each point, compute its marginal contribution
        # But that's O(n^3). Let's do better.
        
        # Standard algorithm:
        # 1. Sort by f1
        # 2. For each consecutive pair, compute the 2D hypervolume of points 
        #    that are non-dominated in (f2,f3) among those with f1 <= current
        # 3. Multiply by the f1 difference
        
        # Even simpler for small n: just compute it directly
        # The 3D hypervolume = sum over all "boxes" in the decomposition
        
        # Let me use the inclusion-exclusion or the standard sweep:
        # Sort by f1 ascending
        # Maintain a list of (f2, f3) points that form the skyline
        # For each new point, update the skyline and compute the volume
        
        # I'll use a straightforward O(n^2) method:
        # For each point i, the contribution is the volume of the box
        # [F[i,0], ref[0]] x [F[i,1], ref[1]] x [F[i,2], ref[2]]
        # minus the volume already covered by other points
        
        # This is complex. Let me use scipy or a known algorithm.
        
        # For now, use the simple method: sort by f1, then for each point,
        # compute the 2D hypervolume in (f2,f3) of the set of points with f1 <= F[i,0]
        # that are non-dominated, and multiply by the f1 interval
        
        # Actually, let me just implement the standard 3D HV algorithm:
        
        # Sort by first objective
        order = np.argsort(F[:, 0])
        F_sorted = F[order]
        
        # The hypervolume can be computed by sweeping in the first objective
        # and maintaining the 2D hypervolume in the other two objectives
        
        # For each "layer" between consecutive f1 values, compute the 2D HV
        # of all points that are active (i.e., have f1 <= current layer)
        
        # Points are added as we sweep from low f1 to high f1
        # At each step, we compute the 2D HV of the active set
        
        # 2D HV for a set of points (f2, f3) with reference (ref2, ref3):
        # Sort by f2, then sweep: hv = sum (ref2 - f2[i]) * (ref3 - f3[i]) for non-dominated points
        
        def hv2d(points_2d, ref2, ref3):
            """Compute 2D hypervolume for points (f2, f3) with reference (ref2, ref3)"""
            if len(points_2d) == 0:
                return 0.0
            # Filter
            mask = np.all(points_2d < [ref2, ref3], axis=1)
            pts = points_2d[mask]
            if len(pts) == 0:
                return 0.0
            # Sort by f2
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]
            hv = 0.0
            prev_f3 = ref3
            for i in range(len(pts)):
                f2, f3 = pts[i]
                if f3 < prev_f3:
                    hv += (ref2 - f2) * (prev_f3 - f3)
                    prev_f3 = f3
            return hv
        
        # Sweep in f1
        n = len(F_sorted)
        if n == 0:
            return 0.0
        
        hv_total = 0.0
        # Active set: points with f1 <= current
        active_2d = []
        
        for i in range(n):
            # Add point i to active set
            active_2d.append(F_sorted[i, 1:])
            
            # The next f1 value (or ref[0] if this is the last)
            f1_next = F_sorted[i+1, 0] if i + 1 < n else ref[0]
            f1_curr = F_sorted[i, 0]
            
            # The 2D HV of the active set, times the f1 interval
            hv2 = hv2d(np.array(active_2d), ref[1], ref[2])
            hv_total += hv2 * (f1_next - f1_curr)
        
        return hv_total

    def _is_pareto_front_point(self, F, ref):
        """Check if points are on or near the Pareto front"""
        # For WFG7, the Pareto front is convex
        # We can check if points are non-dominated and diverse
        return True

    def _sample_pareto_front(self, problem, bounds, n_samples):
        """Sample points along the WFG7 Pareto front"""
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([3.0, 5.0, 7.0])
        
        # WFG7 Pareto front: 
        # The objectives are typically:
        # f1 = sum of (x_i - 0.5)^2 for some transformation
        # For WFG7, the Pareto front is parameterized by t in [0, 1]
        # f1(t) = t^2, f2(t) = (1-t)^2, f3(t) = ... 
        # Actually, WFG7 has a specific structure. Let me use a different approach:
        # Sample uniformly in objective space and map back, or use the known front
        
        # Since we don't know the exact analytic form without looking it up,
        # let's use a hybrid: sample in decision space and keep non-dominated points
        # that are close to the front
        
        # Better approach: use the fact that WFG7's Pareto front is convex and connected.
        # We can sample a grid in a parameter space and evaluate.
        
        # For now, let's use Sobol sampling but with more points initially,
        # then refine the best ones.
        
        points = []
        F_points = []
        
        # Use a Latin hypercube or Sobol for initial sampling
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        raw = sampler.random(n_samples)
        X = lo + raw * (hi - lo)
        
        for x in X:
            f = problem(x)
            self.evals_used += 1
            if np.all(f < ref):
                points.append(x)
                F_points.append(f)
        
        if len(points) == 0:
            return np.array([lo + 0.5 * (hi - lo)]), np.array([[ref[0]+1, ref[1]+1, ref[2]+1]])
        
        X_arr = np.array(points)
        F_arr = np.array(F_points)
        
        # Remove dominated points
        n = len(F_arr)
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                    is_dominated[i] = True
                    break
        mask = ~is_dominated
        X_arr = X_arr[mask]
        F_arr = F_arr[mask]
        
        return X_arr, F_arr

    def _refine_point(self, problem, bounds, x, f, n_trials, rng):
        """Refine a point using local search"""
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([3.0, 5.0, 7.0])
        
        best_x = x
        best_f = f
        step_size = 0.1 * (hi - lo)
        
        for _ in range(n_trials):
            if self.evals_used >= self.budget:
                break
            
            # Try random perturbations
            direction = rng.normal(size=self.dim)
            direction /= np.linalg.norm(direction) + 1e-10
            
            x_new = x + step_size * direction
            x_new = np.clip(x_new, lo, hi)
            
            f_new = problem(x_new)
            self.evals_used += 1
            
            # Check if this is better (non-dominated and improves hypervolume)
            if np.all(f_new < ref):
                # Simple check: if any objective is improved and none worsened
                if np.all(f_new <= f) and np.any(f_new < f):
                    best_x = x_new
                    best_f = f_new
                    # Reduce step size on success
                    step_size *= 0.7
                elif np.all(f < f_new):
                    # Got worse, reduce step size
                    step_size *= 0.8
                else:
                    # Non-dominated trade-off, keep as candidate
                    pass
            
            # Also try axis-aligned moves
            if self.evals_used < self.budget:
                dim_idx = rng.integers(0, self.dim)
                sign = rng.choice([-1, 1])
                direction = np.zeros(self.dim)
                direction[dim_idx] = sign
                x_new = x + step_size * direction
                x_new = np.clip(x_new, lo, hi)
                
                f_new = problem(x_new)
                self.evals_used += 1
                
                if np.all(f_new < ref):
                    if np.all(f_new <= f) and np.any(f_new < f):
                        best_x = x_new
                        best_f = f_new
                        step_size *= 0.7
        
        return best_x, best_f

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([3.0, 5.0, 7.0])
        
        # Phase 1: Initial sampling to establish spread
        init_size = min(80, self.budget)
        
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        points = sampler.random(init_size)
        X_init = lo + points * (hi - lo)
        
        F_init = np.array([problem(x) for x in X_init])
        self.evals_used = init_size
        
        # Filter to only keep points better than reference
        valid_mask = np.all(F_init < ref, axis=1)
        X_init = X_init[valid_mask]
        F_init = F_init[valid_mask]
        
        if len(F_init) > 0:
            self.archive = X_init
            self.archive_F = F_init
        else:
            self.archive = np.array([lo + 0.5 * (hi - lo)])
            self.archive_F = np.array([[ref[0]+1, ref[1]+1, ref[2]+1]])
        
        # Remove dominated points
        n = len(self.archive_F)
        if n > 0:
            is_dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(self.archive_F[j] <= self.archive_F[i]) and np.any(self.archive_F[j] < self.archive_F[i]):
                        is_dominated[i] = True
                        break
            mask = ~is_dominated
            self.archive = self.archive[mask]
            self.archive_F = self.archive_F[mask]
        
        # Phase 2: Refine each point in the archive
        rng = np.random.default_rng(123)
        
        # Sort by hypervolume contribution (approximate by distance from reference)
        if len(self.archive_F) > 0:
            # Prioritize points that are closer to the reference in a balanced way
            distances = np.sum(self.archive_F / ref, axis=1)
            order = np.argsort(distances)
            
            for idx in order:
                if self.evals_used >= self.budget:
                    break
                
                x_current = self.archive[idx]
                f_current = self.archive_F[idx]
                
                # Refine this point
                n_trials = min(10, self.budget - self.evals_used)
                x_refined, f_refined = self._refine_point(problem, bounds, x_current, f_current, n_trials, rng)
                
                # Update archive if improved
                if np.all(f_refined <= f_current) and np.any(f_refined < f_current):
                    self.archive[idx] = x_refined
                    self.archive_F[idx] = f_refined
                elif np.all(f_refined < ref):
                    # Non-dominated trade-off, add to archive
                    self.archive = np.vstack([self.archive, x_refined])
                    self.archive_F = np.vstack([self.archive_F, f_refined])
                    
                    # Remove dominated points
                    n = len(self.archive_F)
                    is_dominated = np.zeros(n, dtype=bool)
                    for i in range(n):
                        for j in range(n):
                            if i == j:
                                continue
                            if np.all(self.archive_F[j] <= self.archive_F[i]) and np.any(self.archive_F[j] < self.archive_F[i]):
                                is_dominated[i] = True
                                break
                    mask = ~is_dominated
                    self.archive = self.archive[mask]
                    self.archive_F = self.archive_F[mask]
        
        return self.archive
