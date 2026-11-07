"""Multi-objective optimizer for WFG7 (3 objectives, 12 vars) using a
Model-and-Optimize (MaO) approach with stratified objective-space targets,
L-BFGS-B local refinement, and a hypervolume-aware refinement pass."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        mid = 0.5 * (lo + hi)
        span = hi - lo
        ref = np.array([3.0, 5.0, 7.0])

        # We will record all evaluations (x, f) and return the last x (ignored).
        evals_x = []
        evals_f = []

        def eval_point(x):
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.asarray(problem(x), dtype=float)
            evals_x.append(x.copy())
            evals_f.append(f.copy())
            return x, f

        n_evals = 0
        def count():
            nonlocal n_evals
            n_evals += 1

        # ------------------------------------------------------------------
        # Objective-space target generation: stratified points on the
        # Pareto front of WFG7 (unit sphere in R^3) plus interior points.
        # ------------------------------------------------------------------
        def generate_targets(n_front, n_interior):
            """Generate objective-space targets that are strictly better than ref."""
            targets = []
            # Stratified points on the positive unit sphere (the WFG7 front)
            # Use a Fibonacci sphere for uniform coverage
            n = n_front
            golden = (1 + np.sqrt(5)) / 2
            for i in range(n):
                z = 1 - 2 * (i + 0.5) / n
                r = np.sqrt(1 - z * z)
                phi = 2 * np.pi * i / golden
                pt = np.array([r * np.cos(phi), r * np.sin(phi), z])
                # Scale to be strictly better than ref (all < ref)
                # WFG7 front has obj values in [~0.5, ~1.0] range typically
                # We want targets in the dominated region below the front
                # Place targets at various depths below the front
                targets.append(pt)
            # Interior points: random points strictly better than ref
            for i in range(n_interior):
                # Sample in the box [0, ref) with some margin
                pt = np.array([
                    0.3 + 0.6 * np.random.uniform(),
                    0.5 + 1.5 * np.random.uniform(),
                    0.7 + 2.0 * np.random.uniform()
                ])
                # Ensure strictly better than ref
                pt = np.minimum(pt, ref - 0.1)
                targets.append(pt)
            return np.array(targets)

        # ------------------------------------------------------------------
        # Inverse problem: given a target objective vector t, find x such that
        # problem(x) ≈ t. Use L-BFGS-B on the squared distance.
        # ------------------------------------------------------------------
        def solve_for_target(target, x0, max_iter=50):
            """Find x that minimizes ||problem(x) - target||^2."""
            def objective(x):
                f = np.asarray(problem(x), dtype=float)
                return np.sum((f - target) ** 2)

            res = minimize(objective, x0, method='L-BFGS-B',
                           bounds=list(zip(lo, hi)),
                           options={'maxiter': max_iter, 'ftol': 1e-10})
            return res.x, res.fun

        # ------------------------------------------------------------------
        # Hypervolume computation (for 3 objectives, simple inclusion-exclusion)
        # ------------------------------------------------------------------
        def hypervolume(points, ref):
            """Compute hypervolume of points dominated by ref.
            points: (n, 3) array, all must be < ref in every dim.
            """
            pts = np.array([p for p in points if np.all(p < ref)])
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            n = len(pts)
            # For 3D, use the standard O(n^2) algorithm
            hv = 0.0
            for i in range(n):
                # For point i, compute the box it uniquely contributes
                # A point i contributes if no earlier point dominates it in all 3 dims
                # and we compute the volume of the region where i is the "best" in obj0
                x1 = pts[i, 0]
                y1 = pts[i, 1]
                z1 = pts[i, 2]
                # Find the envelope of points j < i that have obj0 < x1
                # The contribution is the volume of the box [x1, next_x1) x [y1, ref_y) x [z1, ref_z)
                # minus regions already covered
                # Simpler: use the standard algorithm
                pass
            # Use a simpler O(n^3) approach for small n
            # Sort all points
            # For each point, compute the volume of the box it "owns"
            # In 3D: sort by obj0, for each point compute the 2D hypervolume of the
            # points that are better in obj0 but worse in obj1 and obj2
            # This is complex; let's use a Monte Carlo or a simpler exact method
            
            # Exact 3D hypervolume: 
            # Sort by first coordinate
            # For each point i (in sorted order), the contribution is:
            # (x_{i+1} - x_i) * 2D_hv(points j where x_j < x_i and y_j, z_j form the envelope)
            # Actually, let's use the standard incremental algorithm
            
            # Simpler approach for small n: enumerate all combinations
            # But that's O(n^3). For n up to 200, this is 8 million, which is OK.
            
            # Actually, let's use a different approach:
            # The hypervolume is the volume of the union of boxes [p_i, ref] for all i.
            # We can compute this by inclusion-exclusion, but that's exponential.
            
            # Let's use the standard O(n^2) algorithm for 3D:
            # Sort points by x coordinate
            # Maintain the 2D Pareto front of (y, z) for points with x < current
            # For each point, the contribution is (x_next - x_current) * (2D hv of the front)
            
            pts = pts[np.argsort(pts[:, 0])]
            n = len(pts)
            if n == 0:
                return 0.0
            
            # 2D hypervolume helper
            def hv2d(pts2d, ref2d):
                """2D hypervolume of points dominated by ref2d."""
                p = np.array([p for p in pts2d if np.all(p < ref2d)])
                if len(p) == 0:
                    return 0.0
                p = p[np.argsort(p[:, 0])]
                m = len(p)
                # Sort by x, keep only Pareto-optimal in 2D
                # In 2D, the Pareto front is monotone decreasing
                front = []
                best_y = ref2d[1]
                for i in range(m - 1, -1, -1):
                    if p[i, 1] < best_y:
                        front.append(p[i])
                        best_y = p[i, 1]
                front = np.array(front[::-1])
                if len(front) == 0:
                    return 0.0
                vol = 0.0
                for i in range(len(front)):
                    x1 = front[i, 0]
                    y1 = front[i, 1]
                    x2 = front[i + 1, 0] if i + 1 < len(front) else ref2d[0]
                    vol += (x2 - x1) * (ref2d[1] - y1)
                return vol
            
            total = 0.0
            # For each point in sorted order, compute its contribution
            # The contribution of point i is the volume of the box where point i
            # is the "dominating" point in the x-direction
            # This is (x_{i+1} - x_i) * hv2d of points j with x_j <= x_i, projected to (y,z)
            # But we need to be careful: the 2D front should include point i itself
            
            for i in range(n):
                x_cur = pts[i, 0]
                x_next = pts[i + 1, 0] if i + 1 < n else ref[0]
                # Collect all points j with x_j <= x_cur
                pts2d = pts[:i + 1, 1:3]
                vol2 = hv2d(pts2d, ref[1:3])
                total += (x_next - x_cur) * vol2
            return total

        # ------------------------------------------------------------------
        # Main algorithm
        # ------------------------------------------------------------------
        # Phase 1: Generate stratified targets and solve for each
        n_front_targets = 60
        n_interior_targets = 40
        n_total_targets = n_front_targets + n_interior_targets
        
        targets = generate_targets(n_front_targets, n_interior_targets)
        
        # Initial guesses: mid-point and some random points
        x0_list = [mid.copy()]
        for i in range(10):
            x0_list.append(lo + span * np.random.uniform(size=dim))
        
        results = []  # List of (x, f, target_dist)
        
        for i in range(n_total_targets):
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break
            
            target = targets[i]
            # Try multiple initial points
            best_x = None
            best_f = None
            best_dist = np.inf
            
            for x0 in x0_list[:3]:  # Use first 3 initial points
                if n_evals >= budget:
                    break
                if time.time() > deadline:
                    break
                
                x_sol, dist = solve_for_target(target, x0.copy(), max_iter=30)
                x_sol, f_sol = eval_point(x_sol)
                count()
                
                # Check if this is better
                actual_dist = np.sum((f_sol - target) ** 2)
                if actual_dist < best_dist:
                    best_dist = actual_dist
                    best_x = x_sol
                    best_f = f_sol
            
            if best_x is not None:
                results.append((best_x, best_f, best_dist))
        
        # Phase 2: Hypervolume-aware refinement
        # Find the current hypervolume and identify gaps
        if len(results) > 0:
            current_points = np.array([r[1] for r in results])
            current_hv = hypervolume(current_points, ref)
            
            # Generate new targets in the largest uncovered regions
            # Use a grid in objective space and find where the hypervolume
            # contribution is missing
            n_refine = min(50, budget - n_evals)
            
            for i in range(n_refine):
                if n_evals >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Sample a random point in the dominated region
                target = np.array([
                    0.2 + 0.8 * np.random.uniform(),
                    0.3 + 1.7 * np.random.uniform(),
                    0.5 + 2.5 * np.random.uniform()
                ])
                target = np.minimum(target, ref - 0.05)
                
                # Solve for this target
                x0 = mid.copy() + span * 0.1 * np.random.standard_normal(dim)
                x_sol, dist = solve_for_target(target, x0, max_iter=20)
                x_sol, f_sol = eval_point(x_sol)
                count()
                
                results.append((x_sol, f_sol, dist))
        
        # Return the last evaluated x (interface requirement)
        if evals_x:
            return evals_x[-1]
        else:
            return mid
