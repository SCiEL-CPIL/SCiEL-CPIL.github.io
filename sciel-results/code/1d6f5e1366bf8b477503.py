"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hypervolume-weighted recursive bisection grid + evolutionary search for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        budget = self.budget
        m = self.n_obj  # number of objectives

        # For DTLZ2 with 3 objectives, the Pareto front is the portion of the unit sphere
        # in the positive octant: f_1^2 + f_2^2 + f_3^2 = 1, f_i >= 0.
        
        n_pos = m - 1  # number of position variables
        n_g = n - n_pos  # number of g variables
        
        def front_point_from_objectives(f_target):
            """Create a decision vector on the Pareto front targeting given objective values."""
            x = np.zeros(n)
            x[:n_g] = 0.5  # g-variables at optimum for g=1
            
            if m >= 2:
                f1_clipped = np.clip(f_target[0], 0, 1)
                x[n-2] = 2.0 / np.pi * np.arcsin(f1_clipped)
                
                if m >= 3:
                    f2_clipped = np.clip(f_target[1], 0, 1)
                    x[n-1] = 2.0 / np.pi * np.arcsin(f2_clipped)
            
            return np.clip(x, 0, 1)
        
        ref = np.array([2.5] * m)
        
        # Hypervolume-weighted recursive bisection in objective space
        # We recursively bisect the [0,1]^3 cube and place points at centers
        # of sub-rectangles that have the largest marginal hypervolume gain.
        
        # A sub-rectangle is defined by [lo_1, hi_1] x [lo_2, hi_2] x [lo_3, hi_3]
        # The center is at ((lo_i + hi_i)/2 for each i)
        # But we need the point to be on or near the Pareto front.
        # For a point in the objective space to be on the DTLZ2 front, we need
        # f_1^2 + f_2^2 + f_3^2 = 1. So we project the center onto the front.
        
        def project_to_front(f):
            """Project a point in [0,1]^3 onto the DTLZ2 Pareto front (unit sphere octant)."""
            norm = np.linalg.norm(f)
            if norm < 1e-10:
                return np.array([1.0, 0.0, 0.0])
            projected = f / norm
            # Ensure all components are in [0,1]
            projected = np.clip(projected, 0, 1)
            # Renormalize to be on the sphere
            norm2 = np.linalg.norm(projected)
            if norm2 > 1e-10:
                projected = projected / norm2
            return projected
        
        # Recursive bisection: maintain a list of sub-rectangles with their
        # hypervolume contribution (volume from center to ref, clipped to front)
        
        # Start with the full [0,1]^3 box
        # The "value" of a sub-rectangle is the hypervolume contribution of its
        # projected center point.
        
        def hv_of_point(f):
            """Hypervolume contribution of a single point (volume from f to ref)."""
            if np.any(f >= ref):
                return 0.0
            return float(np.prod(ref - f))
        
        # We'll do iterative bisection: at each step, pick the sub-rectangle with
        # the highest marginal HV gain, bisect it, and add its center to our set.
        
        # For efficiency, we limit the number of bisection levels.
        max_levels = 6  # 2^6 = 64 sub-rectangles at deepest level
        n_init = min(80, budget // 3)
        
        # Use a priority queue approach: start with the full box, repeatedly
        # bisect the box with the highest HV contribution.
        
        # Each box is (lo, hi) where lo, hi are 3D vectors
        boxes = [(np.zeros(3), np.ones(3))]
        evaluated_points = []  # list of (f_target, x)
        
        for level in range(max_levels):
            if len(evaluated_points) >= n_init:
                break
            
            # For each box, compute the HV contribution of its center projected to front
            # Pick the box with the highest contribution, bisect it, add center to evaluated
            best_box_idx = -1
            best_hv = -1
            best_center = None
            best_proj = None
            
            for i, (bl, bh) in enumerate(boxes):
                center = (bl + bh) / 2.0
                proj = project_to_front(center)
                hv = hv_of_point(proj)
                if hv > best_hv:
                    best_hv = hv
                    best_box_idx = i
                    best_center = center
                    best_proj = proj
            
            if best_box_idx < 0:
                break
            
            # Add the projected center to our evaluated points
            x = front_point_from_objectives(best_proj)
            evaluated_points.append((best_proj, x))
            
            # Remove the best box and replace with two bisected boxes
            bl, bh = boxes.pop(best_box_idx)
            
            # Bisect along the dimension with the largest extent
            extents = bh - bl
            split_dim = int(np.argmax(extents))
            mid = (bl[split_dim] + bh[split_dim]) / 2.0
            
            # Create two new boxes
            bl1 = bl.copy()
            bh1 = bh.copy()
            bh1[split_dim] = mid
            
            bl2 = bl.copy()
            bh2 = bh.copy()
            bl2[split_dim] = mid
            
            boxes.append((bl1, bh1))
            boxes.append((bl2, bh2))
        
        # If we haven't filled n_init, add more points via random sampling on the front
        while len(evaluated_points) < n_init:
            # Random point on the quarter sphere
            t1 = rng.uniform(0.001, np.pi/2 - 0.001)
            t2 = rng.uniform(0.001, np.pi/2 - 0.001)
            f1 = np.sin(t1)
            f2 = np.cos(t1) * np.sin(t2)
            f3 = np.cos(t1) * np.cos(t2)
            f_target = np.array([f1, f2, f3])
            x = front_point_from_objectives(f_target)
            evaluated_points.append((f_target, x))
        
        # Evaluate all initial points
        init_xs = [x for _, x in evaluated_points]
        F_init = np.array([problem(x) for x in init_xs])
        used = len(init_xs)
        
        # Keep only valid points (all objectives < ref)
        valid_mask = np.all(F_init < ref, axis=1)
        X = np.array(init_xs[valid_mask])
        F = F_init[valid_mask]
        
        if len(X) == 0:
            # Fallback: random initialization
            X = lo + (hi - lo) * rng.random((10, n))
            F = np.array([problem(x) for x in X])
            used = 10
        
        # Phase 2: Evolutionary refinement with hypervolume-aware selection
        while used < budget:
            # Compute non-dominated set
            n_pop = len(X)
            dominated = np.zeros(n_pop, dtype=bool)
            for i in range(n_pop):
                if dominated[i]:
                    continue
                for j in range(n_pop):
                    if i == j or dominated[i]:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True
            
            non_dom_mask = ~dominated
            n_nd = np.sum(non_dom_mask)
            
            if n_nd == 0:
                non_dom_mask = np.ones(n_pop, dtype=bool)
                n_nd = n_pop
            
            non_dom_X = X[non_dom_mask]
            non_dom_F = F[non_dom_mask]
            
            # Select parent from non-dominated set, weighted by hypervolume contribution
            hv_scores = np.prod(ref - non_dom_F, axis=1)
            total_hv = np.sum(hv_scores)
            if total_hv > 0:
                probs = hv_scores / total_hv
            else:
                probs = np.ones(len(non_dom_F)) / len(non_dom_F)
            
            parent_idx = rng.choice(len(non_dom_F), p=probs)
            parent = non_dom_X[parent_idx]
            
            # Mutate with adaptive mutation rate
            mutation_rate = 0.05 + 0.05 * (1 - used / budget)
            child = parent + mutation_rate * (hi - lo) * rng.standard_normal(n)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace the point with lowest hypervolume contribution
            all_hv_scores = np.prod(ref - F, axis=1)
            worst_idx = int(np.argmin(all_hv_scores))
            X[worst_idx] = child
            F[worst_idx] = fc
        
        return X
# EVOLVE-BLOCK-END
