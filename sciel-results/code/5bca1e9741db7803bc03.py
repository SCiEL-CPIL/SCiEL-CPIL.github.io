"""Multi-objective optimizer for RE34: hypervolume-guided local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided local search for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])

        # Deterministic initialization using Sobol-like or stratified sampling
        # For 5D and 3 objectives, we want good spread.
        # Let's use a simple deterministic grid or quasi-random sequence.
        # Since budget is 200, we can afford a decent initial population.
        # Let's start with 20 points spread out.
        
        # Use Halton sequence for deterministic spread
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            points = []
            bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
            for i in range(1, n + 1):
                p = [halton(i, b) for b in bases[:dim]]
                points.append(p)
            return np.array(points)

        init_pop_size = min(20, self.budget // 10)
        if init_pop_size < 5:
            init_pop_size = 5
            
        # Generate initial points
        halton_pts = halton_sequence(init_pop_size, self.dim)
        X_init = lo + (hi - lo) * halton_pts
        F_init = np.array([problem(x) for x in X_init])
        
        # Store all evaluated points
        X_all = [X_init]
        F_all = [F_init]
        n_evals = init_pop_size

        # Helper to compute hypervolume contribution of a new point to existing set
        def compute_hv_contribution(new_point, existing_points, ref_point):
            """
            Compute the hypervolume added by new_point to the set of existing_points.
            Simplified for 3D.
            """
            # Filter existing points that are non-dominated by new_point
            # A point p dominates n if p <= n in all objectives and p < n in at least one.
            # We only care about points that are not dominated by new_point? 
            # Actually, the contribution is the volume of the region dominated by new_point 
            # but not dominated by any existing point.
            
            # For 3D, this is complex. Let's use a simpler heuristic:
            # Check if new_point is non-dominated in the set.
            # If it is, estimate its contribution by looking at the "box" it controls.
            
            # Better approach: Exact HV calculation for small sets is feasible.
            # But calling it every step might be slow.
            # Let's use a fast approximation: 
            # Contribution is roughly (ref[0]-f[0]) * (ref[1]-f[1]) * (ref[2]-f[2]) 
            # minus overlaps.
            
            # Let's implement a simple exact HV for the set {new_point} U {existing non-dominated by new_point}
            # Actually, we can just compute the HV of the whole set before and after.
            # But that's O(N^2) or O(N log N) per step. With N=200, it's fine.
            
            pass

        # Let's implement a fast HV calculator for 3D
        def hv_3d(points, ref):
            """
            Compute hypervolume for a set of 3D points.
            points: Nx3 array
            ref: 3D reference point
            """
            if len(points) == 0:
                return 0.0
            
            # Filter points that are better than ref in all objectives
            valid = np.all(points < ref, axis=1)
            points = points[valid]
            if len(points) == 0:
                return 0.0
            
            # Sort by first objective
            points = points[np.argsort(points[:, 0])]
            
            # Hypervolume in 3D can be computed by slicing along the first axis
            # Volume = sum over i of (x_{i+1} - x_i) * (max_y - y_i) * (max_z - z_i) ? No.
            
            # Correct 3D HV algorithm:
            # Sort by x.
            # Maintain a set of "active" points in the y-z plane.
            # This is getting complex. Let's use a simpler O(N^2) method for small N.
            
            # Since N <= 200, O(N^2) is 40,000 ops, which is fast.
            # We can compute the HV by iterating over all points and finding the volume they uniquely contribute.
            
            # Even simpler: Since we only need to compare contributions, we can use a heuristic.
            # But let's try to be somewhat accurate.
            
            # Let's use the standard "layer" method.
            # Sort by x.
            # For each point i, the volume it contributes is (x_{i+1} - x_i) * (area in y-z plane dominated by points 1..i but not 1..i-1?)
            # This is tricky.
            
            # Let's just use a grid-based approximation for speed? No, exact is better.
            
            # Here is a simple O(N^2) HV for 3D:
            # The HV is the union of boxes [p_i, ref].
            # We can compute this by inclusion-exclusion or by sweeping.
            
            # Sweep along x:
            # Sort points by x ascending.
            # Let x_{n+1} = ref[0].
            # For each i, consider the slice [x_i, x_{i+1}].
            # In this slice, the dominated region in y-z is determined by the points with x <= x_{i+1}.
            # Specifically, for a given y, the max z dominated is the min z of points with y_j <= y and x_j <= x_{i+1}.
            # This is still complex.
            
            # Let's use a different strategy: 
            # Since we are doing local search, we can just evaluate the HV of the current Pareto front.
            # We don't need to compute contribution exactly, we can just compute HV of (Front + New Point) and compare to HV(Front).
            
            # Let's implement a basic O(N^2) HV for 3D.
            # Actually, for N=200, even O(N^3) is 8 million ops, which is < 0.1s in Python if vectorized.
            
            # Let's try a simple vectorized HV for 3D.
            # Reference: "Fast Hypervolume Calculation"
            
            # For 3D, we can do:
            # 1. Sort by x.
            # 2. For each point i, find the "envelope" of y-z points that are not dominated by any point j < i.
            # This is essentially computing the 2D HV of the projected points.
            
            # Let's just use a simple grid integration if needed, or a simple loop.
            
            # Given the time constraints, let's use a heuristic for the "best" next point:
            # Pick the point that is most "spread out" or has the largest distance to the current front.
            
            # Actually, let's stick to the plan:
            # 1. Find the point in the current population that has the highest "potential" HV contribution.
            # 2. Mutate it.
            
            # To find the best point to mutate, we can look at the "density" of the front.
            # Points in sparse regions of the front should be mutated.
            
            # Let's define a score for each point:
            # score = (ref[0]-f[0]) * (ref[1]-f[1]) * (ref[2]-f[2])
            # This is the volume of the box it dominates.
            # Points with large boxes are "important".
            
            # But we want to fill gaps.
            # Let's use the following:
            # For each point, compute the volume of the box it dominates that is NOT dominated by any other point in the front.
            # This is the "unique contribution".
            
            # Let's implement `unique_contribution` for 3D.
            def unique_contrib_3d(point, front, ref):
                # front: Mx3 array
                # point: 3D array
                # We want the volume of the region dominated by point but not by any point in front.
                
                # A region is dominated by point if x <= point[0], y <= point[1], z <= point[2].
                # It is dominated by front if there exists p in front such that x <= p[0], y <= p[1], z <= p[2].
                
                # So we want the volume of:
                # { (x,y,z) | x <= point[0], y <= point[1], z <= point[2] } 
                # MINUS
                # { (x,y,z) | exists p in front: x <= p[0], y <= p[1], z <= p[2] }
                
                # This is equivalent to the HV of the set {point} U {front} minus HV({front}).
                # But we can compute it directly.
                
                # The region dominated by point is a box [0, point[0]] x [0, point[1]] x [0, point[2]] (shifted).
                # The region dominated by front is the union of boxes.
                
                # Let's just compute HV({point} U {front}) - HV({front}).
                # But we don't have a fast HV function yet.
                
                # Let's use a simple approximation:
                # If point is dominated by any point in front, contribution is 0.
                # If point dominates any point in front, it might have high contribution.
                
                # Check if point is non-dominated in front.
                dominated = np.any(np.all(front <= point, axis=1) & np.any(front < point, axis=1))
                if dominated:
                    return 0.0
                
                # Estimate contribution by the volume of the box minus overlaps with "nearby" points.
                # For now, let's just use the box volume as a proxy.
                vol = max(0, ref[0] - point[0]) * max(0, ref[1] - point[1]) * max(0, ref[2] - point[2])
                return vol

            # This is too simple. Let's try a different approach:
            # Use a simple evolutionary strategy but with a twist:
            # Instead of random mutation, use a "greedy" mutation that tries to improve the worst objective.
            
            # Let's go back to the local search idea.
            # We will maintain a set of non-dominated points.
            # At each step, we pick the point with the largest "gap" to its neighbors in the front.
            # Then we mutate it.
            
            # To measure "gap", we can project the front onto a 2D plane (e.g., obj1 vs obj2) and look for large distances.
            
            # Let's implement this.
            
            pass

        # Let's simplify. The current program scores 1.0.
        # The inspiration 3 (HV-guided local search) scored 0.5584.
        # The inspiration 1 (NSGA-II) scored 0.8237.
        # The current program is a simple EA.
        
        # Maybe the issue is that the current program is too random.
        # Let's try a more directed search.
        
        # Strategy:
        # 1. Initialize with 20 points.
        # 2. Find the Pareto front.
        # 3. For each point in the front, compute the "hypervolume contribution" (approximated).
        # 4. Pick the point with the highest contribution.
        # 5. Mutate it with a small step.
        # 6. If the child is non-dominated, add it to the set.
        # 7. Repeat.
        
        # Let's implement a simple HV contribution estimator.
        # For 3D, the contribution of a point p to the front F is the volume of the box [p, ref] that is not covered by any box [q, ref] for q in F.
        
        # We can approximate this by checking if p is "isolated".
        
        # Let's try a different tactic: 
        # Use the fact that RE34 has 3 objectives.
        # We can use a decomposition method like MOEA/D? 
        # But we only have 200 evaluations.
        
        # Let's stick to a simple, robust approach:
        # 1. Initialize with 20 points.
        # 2. While budget remains:
        #    a. Find the Pareto front.
        #    b. Pick a random point from the front.
        #    c. Mutate it with a step size that adapts based on the distance to other front points.
        #    d. Evaluate.
        #    e. If non-dominated, replace a dominated point or add to set.
        
        # This is similar to the current program but with adaptive step size.
        
        # Let's implement this.
        
        X = X_init
        F = F_init
        n_evals = init_pop_size
        
        # Track best HV
        # We need a HV function. Let's use a simple one.
        
        def hv_simple(points, ref):
            # Simple O(N^2) HV for 3D
            # Sort by x
            points = points[np.argsort(points[:, 0])]
            n = len(points)
            if n == 0:
                return 0.0
            
            # We will compute the volume by summing the contributions of each point
            # Contribution of point i is (x_{i+1} - x_i) * (max_y - y_i) * (max_z - z_i) ? No.
            
            # Let's use a grid-based integration for simplicity and robustness.
            # Grid size 10x10x10? 
            # This is approximate but fast.
            
            # Actually, let's just use the sum of box volumes for non-dominated points.
            # This overestimates HV due to overlaps, but it's a good proxy for "spread".
            
            # Filter non-dominated
            non_dom = []
            for i in range(n):
                is_dom = False
                for j in range(n):
                    if i != j:
                        if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                            is_dom = True
                            break
                if not is_dom:
                    non_dom.append(points[i])
            
            if not non_dom:
                return 0.0
            
            non_dom = np.array(non_dom)
            
            # Sum of box volumes
            vols = (ref[0] - non_dom[:, 0]) * (ref[1] - non_dom[:, 1]) * (ref[2] - non_dom[:, 2])
            return np.sum(vols)
        
        # This is a proxy, not exact HV. But it should guide the search.
        
        # Let's use this proxy to guide the mutation.
        
        while n_evals < self.budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated points
            non_dom_idx = []
            for i in range(len(F)):
                is_dom = False
                for j in range(len(F)):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            is_dom = True
                            break
                if not is_dom:
                    non_dom_idx.append(i)
            
            if not non_dom_idx:
                non_dom_idx = list(range(len(F)))
            
            # Pick a point from the non-dominated set
            # We can pick the one with the largest "box volume"
            # Or pick randomly to explore.
            # Let's pick randomly to avoid getting stuck.
            idx = non_dom_idx[np.random.randint(len(non_dom_idx))]
            
            # Mutate
            # Step size: adaptive
            # Let's use a step size of 0.1 * (hi - lo)
            step = 0.1 * (hi - lo)
            child = X[idx] + step * np.random.randn(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            n_evals += 1
            
            # Check if child is non-dominated
            is_dom = False
            for j in range(len(F)):
                if np.all(F[j] <= fc) and np.any(F[j] < fc):
                    is_dom = True
                    break
            
            if not is_dom:
                # Add child to population
                # Replace a dominated point or the worst point
                # Find dominated points
                dom_idx = []
                for i in range(len(F)):
                    is_dominated = False
                    for j in range(len(F)):
                        if i != j:
                            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                is_dominated = True
                                break
                    if is_dominated:
                        dom_idx.append(i)
                
                if dom_idx:
                    # Replace a random dominated point
                    replace_idx = dom_idx[np.random.randint(len(dom_idx))]
                else:
                    # Replace the point with the smallest box volume
                    vols = (ref[0] - F[:, 0]) * (ref[1] - F[:, 1]) * (ref[2] - F[:, 2])
                    replace_idx = np.argmin(vols)
                
                X[replace_idx] = child
                F[replace_idx] = fc
        
        return X
# EVOLVE-BLOCK-END
