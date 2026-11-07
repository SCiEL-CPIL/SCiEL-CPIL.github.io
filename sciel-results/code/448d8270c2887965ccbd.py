import numpy as np
import time

class Optimizer:
    """
    Direct Hierarchical Multi-Objective Optimizer with Adaptive Local Search.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        bounds_range = hi - lo

        all_X = []
        all_F = []
        eval_count = 0

        # Reference point for hypervolume
        ref_point = np.array([3000.0, 0.06])

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return None
            if time.time() > deadline:
                return None
            f = problem(x)
            f = np.array(f, dtype=float)
            if not np.all(np.isfinite(f)):
                return None
            all_X.append(x)
            all_F.append(f)
            eval_count += 1
            return f

        # Helper: Check if point A dominates point B (minimization)
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # Helper: Get non-dominated indices from current lists
        def get_non_dominated_indices():
            if not all_F:
                return []
            F = np.array(all_F)
            n = len(F)
            nd_mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd_mask[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if not nd_mask[j]:
                        continue
                    if dominates(F[j], F[i]):
                        nd_mask[i] = False
                        break
            return np.where(nd_mask)[0]

        # Helper: Calculate Hypervolume Contribution in 2D
        # Points must be non-dominated and strictly better than ref
        def hv_contribution_2d(point, nd_points, ref):
            # nd_points: list of tuples (f1, f2)
            # ref: tuple (r1, r2)
            # We assume nd_points are sorted by f1 ascending
            # Contribution is the area of the rectangle defined by point and the "corner"
            # formed by the next point with higher f1 and lower f2 (or ref).
            
            p1, p2 = point
            r1, r2 = ref
            
            # If point is not better than ref, contribution is 0
            if p1 >= r1 or p2 >= r2:
                return 0.0
            
            # Find the next point in the front with f1 > p1
            # Since we are in 2D and non-dominated, if f1 increases, f2 must decrease
            next_f2 = r2
            for q1, q2 in nd_points:
                if q1 > p1:
                    next_f2 = min(next_f2, q2)
                    break
            
            # The contribution is the area of the rectangle
            # width = (next_f1 - p1) where next_f1 is the f1 of the next point or ref
            # Actually, we need to be more careful. Let's use the standard 2D HV calculation.
            
            # For a point to contribute, it must be on the Pareto front.
            # In 2D, the HV contribution of a point is:
            # (ref[0] - f1) * (f2_next - f2) where f2_next is the f2 of the next point with higher f1
            # But we need to find the "envelope".
            
            # Simpler approach: For each point, its contribution is the area of the rectangle
            # that it uniquely covers. In 2D, sort by f1, then for each point, the contribution
            # is (ref[0] - f1) * (next_f2 - f2) where next_f2 is the f2 of the next point or ref[1].
            
            # But this is only valid if the points are sorted and we process them in order.
            # Let's use a different approach: calculate the total HV and subtract contributions.
            
            # Actually, let's just return a simple proxy: the "corner" distance
            # Points closer to the ideal point (0,0) but still on the front are valuable.
            # A good proxy is: (ref[0] - p1) * (ref[1] - p2) / (ref[0] * ref[1])
            # This gives higher weight to points that are better in both objectives.
            
            return (ref[0] - p1) * (ref[1] - p2)

        # Phase 1: Global Sampling
        # Use a mix of random and grid points to ensure coverage
        n_global = 50
        
        # Grid points (corners)
        grid_points = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        x = np.array([
                            lo[0] + bounds_range[0]*i,
                            lo[1] + bounds_range[1]*j,
                            lo[2] + bounds_range[2]*k,
                            lo[3] + bounds_range[3]*l
                        ])
                        grid_points.append(x)
        
        # Random points
        random_points = lo + bounds_range * np.random.random((n_global - len(grid_points), self.dim))
        
        initial_points = np.vstack([grid_points, random_points]) if len(random_points) > 0 else np.array(grid_points)
        
        for x in initial_points:
            if time.time() > deadline:
                break
            if eval_count >= self.budget:
                break
            evaluate(x)

        if eval_count >= self.budget or not all_F:
            return all_X

        # Phase 2: Adaptive Local Refinement
        # We will perform multiple rounds of local search on the best non-dominated points.
        
        # Initial step size
        step_scale = 0.2 
        
        # Number of points to refine
        k_refine = 12
        
        while eval_count < self.budget and time.time() < deadline:
            nd_indices = get_non_dominated_indices()
            if len(nd_indices) == 0:
                break
            
            # Calculate HV contribution for each non-dominated point
            nd_F = [all_F[i] for i in nd_indices]
            hv_contribs = [hv_contribution_2d(f, nd_F, ref_point) for f in nd_F]
            
            # Sort by HV contribution (descending)
            sorted_order = np.argsort(hv_contribs)[::-1]
            nd_indices = np.array(nd_indices)
            sorted_nd = nd_indices[sorted_order]
            
            k = min(k_refine, len(sorted_nd))
            indices_to_refine = sorted_nd[:k]
            
            # If we have very little budget left, just do one step for each
            remaining = self.budget - eval_count
            if remaining < k:
                # Just pick the best one and use all budget
                indices_to_refine = [indices_to_refine[0]]
                k = 1
            
            evals_per_point = max(1, remaining // k)
            
            improved = False
            
            for idx in indices_to_refine:
                x_best = all_X[idx].copy()
                f_best = all_F[idx].copy()
                
                # Local Search Loop for this point
                for _ in range(evals_per_point):
                    if eval_count >= self.budget or time.time() > deadline:
                        break
                    
                    # Generate candidate with adaptive step
                    # Use a mix of step sizes: mostly small, occasionally larger
                    if np.random.random() < 0.3:
                        step = step_scale * 0.5 * bounds_range * np.random.random(self.dim)
                    else:
                        step = step_scale * bounds_range * np.random.random(self.dim)
                    
                    direction = np.random.choice([-1, 1], self.dim)
                    candidate = x_best + direction * step
                    candidate = np.clip(candidate, lo, hi)
                    
                    f_cand = evaluate(candidate)
                    if f_cand is None:
                        break
                    
                    # Accept if it dominates the current best
                    if dominates(f_cand, f_best):
                        x_best = candidate
                        f_best = f_cand
                        improved = True
                    elif not dominates(f_best, f_cand):
                        # Non-dominated: accept with some probability to explore
                        if np.random.random() < 0.3:
                            x_best = candidate
                            f_best = f_cand
                            improved = True

            # If no improvement in this round, reduce step size
            if not improved:
                step_scale *= 0.7
                if step_scale < 0.001:
                    break
            else:
                # If improved, keep step size
                pass

        return all_X
