"""Multi-objective optimizer for DTLZ2 using structure-aware LHS + adaptive mutation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _hypervolume_3d(points, ref):
    """Compute 3D hypervolume dominated by points w.r.t. reference point.
    points: (N, 3) array, ref: (3,) array.
    Only points strictly better than ref in all dims contribute.
    """
    # Filter points strictly better than ref
    mask = np.all(points < ref, axis=1)
    pts = points[mask]
    n = len(pts)
    if n == 0:
        return 0.0
    if n == 1:
        return float(np.prod(ref - pts[0]))
    
    # Sort by first objective
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    
    # For 3D HV: sort by x, sweep with 2D HV of (y,z)
    # 2D HV of a set of points w.r.t. ref (ry, rz):
    # sort by y, keep min z seen so far
    total = 0.0
    prev_x = ref[0]
    # We'll sweep: for each point in x-order, the volume it contributes
    # is (x_prev - x_i) * 2D_HV of points with index <= i in (y,z)
    # But we need to be careful: the 2D HV changes as we add points.
    # Alternative: use the standard 3D HV algorithm.
    
    # Standard approach: sort by x, then for each consecutive pair,
    # compute 2D HV of the "active" set.
    # Active set for x in [x_i, x_{i+1}] is all points with x <= x_i.
    
    # More efficient: use the fact that in 3D, HV = sum over i of
    # (x_{i-1} - x_i) * HV_2d({(y_j, z_j) : j <= i})
    # where HV_2d is computed on the 2D projection.
    
    # Let's use a cleaner algorithm:
    # Sort by x. For each i from 0 to n-1:
    #   width_i = (x_{i-1} - x_i) where x_{-1} = ref[0]
    #   The 2D HV of the set {(y_j, z_j) for j=0..i} w.r.t. (ref[1], ref[2])
    #   contributes width_i * that_2d_hv
    
    # But this double-counts. The correct formula:
    # HV = sum_{i=0}^{n-1} (x_{i-1} - x_i) * HV_2d(S_i)
    # where S_i = {(y_j, z_j) : j = 0, ..., i} and x_{-1} = ref[0].
    # Wait, that's not right either. Let me think again.
    
    # Correct 3D HV via sweep:
    # Sort points by x ascending.
    # For the slab between x_{i-1} and x_i (x_{-1}=ref[0]), the cross-section
    # at any x in (x_i, x_{i-1}) is the 2D region dominated by all points
    # with x <= x_{i-1}... no.
    
    # Let me use a different well-known algorithm for 3D HV:
    # 1. Sort by x.
    # 2. For each point i, compute the 2D HV of the set of points j where x_j <= x_i
    #    projected to (y,z), but only counting the contribution of point i's
    #    "new" region.
    
    # Actually, the simplest correct approach for small N:
    # Use inclusion-exclusion or the standard sweep.
    
    # Let me just use the standard 3D HV algorithm:
    # Sort by x. Maintain a list of (y, z) for points seen so far.
    # For each new point, the volume it adds is:
    # (x_prev - x_new) * 2D_HV_contribution_new_point
    # where 2D_HV_contribution is the area in (y,z) that is dominated by
    # the new point but not by any previously seen point.
    
    # 2D HV of a single point (y, z) w.r.t. ref (ry, rz) is (ry-y)*(rz-z).
    # The 2D HV of a set is computed by sorting by y and taking the envelope.
    
    # For the incremental approach: when adding point (y_new, z_new),
    # the new 2D area it contributes is the area in the rectangle
    # [y_new, ry] x [z_new, rz] that is NOT covered by the existing 2D HV envelope.
    
    # The existing envelope: sort existing points by y, the upper envelope of z
    # (for minimization, we want the minimum z for each y-range).
    
    # This is getting complex. For N <= 200, let me use a simpler O(N^2) or O(N^3) method.
    
    # Simple O(N^2) 3D HV:
    # Sort by x. For each point i, compute the 2D HV of the "box" it uniquely covers.
    # A point i covers the box [x_i, x_{i-1}] x [y_i, y_min_above_i] x [z_i, z_min_above_i]
    # where y_min_above_i and z_min_above_i are determined by other points.
    
    # Let me just implement the standard algorithm properly.
    
    # Algorithm for 3D HV (from literature):
    # 1. Sort points by x ascending.
    # 2. Initialize a list for the 2D envelope.
    # 3. For each point in order, compute the 2D area it adds to the envelope,
    #    multiply by (x_prev - x_current), add to total.
    
    # 2D envelope maintenance:
    # The 2D HV of a set of points w.r.t. (ry, rz) can be computed by:
    # Sort by y, then the HV is sum over consecutive y-values of
    # (y_{i-1} - y_i) * (rz - z_min_so_far)
    # where z_min_so_far is the minimum z among points with y <= y_{i-1}.
    
    # For incremental 2D HV: when adding a new point (y_new, z_new),
    # the additional 2D area is the area in [y_new, ry] x [z_new, rz]
    # that is not already covered.
    
    # The existing 2D envelope is a step function: for each y-interval,
    # there's a z-value (the min z of points with y in that range or lower).
    
    # Let me maintain the envelope as a sorted list of (y, z) breakpoints
    # where z is the minimum z achieved up to that y.
    
    # Actually, for simplicity and correctness with N <= 200, let me use
    # a direct O(N^2) method:
    
    # For each point i, its contribution to HV is the volume of the box
    # [x_i, x_{i-1}] x [y_i, y_i'] x [z_i, z_i']
    # where x_{i-1} is the x of the previous point (or ref[0] if i=0),
    # y_i' = min(ref[1], min{y_j : j != i, x_j <= x_{i-1}, y_j < y_i})... 
    # This is getting complicated.
    
    # Let me use a different approach: compute HV by summing over cells.
    # For small N, I can use the following:
    # HV = sum over all subsets... no, that's exponential.
    
    # OK let me just implement the standard sweep-line algorithm for 3D HV.
    
    # Sort by x
    pts_sorted = pts[np.argsort(pts[:, 0])]
    
    total_hv = 0.0
    # We'll maintain the 2D HV envelope incrementally
    # The 2D envelope is represented as a list of (y, z) points that form
    # the "staircase" of minimum z for each y-range.
    
    # For the 2D part: given a set of (y, z) points, the 2D HV w.r.t. (ry, rz) is:
    # Sort by y. The HV is the area under the step function where at each y,
    # the height is (rz - min_z_up_to_y).
    
    # Let me maintain a list of (y, z) breakpoints for the 2D envelope.
    # The envelope is: for y in [0, ry], the min z among points with y_coord <= y.
    # This is a non-increasing step function.
    
    # I'll maintain it as a sorted list of (y_break, z_value) where z_value
    # is the min z for y >= y_break up to the next breakpoint.
    
    # Actually, let me just recompute 2D HV from scratch at each step since N is small.
    
    def _hv_2d(points_2d, ref_2d):
        """Compute 2D HV of points w.r.t. ref. points_2d: (M, 2), ref_2d: (2,)"""
        m = len(points_2d)
        if m == 0:
            return 0.0
        # Sort by first coordinate
        order = np.argsort(points_2d[:, 0])
        p = points_2d[order]
        total = 0.0
        min_z = ref_2d[1]  # current minimum z
        prev_y = ref_2d[0]
        for i in range(m):
            y_i, z_i = p[i]
            if z_i < min_z:
                # The region from prev_y to y_i has height (ref_2d[1] - min_z)
                total += (prev_y - y_i) * (ref_2d[1] - min_z)
                min_z = z_i
            prev_y = y_i
        # Final segment
        total += (prev_y - ref_2d[0]) * (ref_2d[1] - min_z)
        # Wait, this isn't right. Let me redo.
        
        # The 2D HV: sort by y. For each point, the width it contributes is
        # (y_prev - y_current) and the height is (rz - min_z_so_far).
        # min_z_so_far is the minimum z among all points with y <= y_prev.
        
        total = 0.0
        min_z = ref_2d[1]
        prev_y = ref_2d[0]
        for i in range(m):
            y_i, z_i = p[i]
            # The slab from y_i to prev_y has height (rz - min_z)
            total += (prev_y - y_i) * (ref_2d[1] - min_z)
            if z_i < min_z:
                min_z = z_i
            prev_y = y_i
        return total
    
    # Now for 3D:
    # Sort by x. For each consecutive pair (x_i, x_{i-1}), the 2D HV of
    # all points with x <= x_{i-1} contributes (x_{i-1} - x_i) * that_2d_hv.
    # But we need to be careful: the 2D HV should be computed on the projection
    # of all points with x <= x_{i-1}.
    
    # Let me do it incrementally:
    # Sort by x ascending.
    # Maintain a list of (y, z) for all points seen so far.
    # For each new point, before adding it, compute the 2D HV of the current set.
    # The contribution is (x_prev - x_new) * 2D_HV.
    # Then add the new point to the set.
    
    # Wait, that's not quite right. Let me think more carefully.
    
    # The 3D HV is the volume of the union of boxes [x_i, ref[0]] x [y_i, ref[1]] x [z_i, ref[2]].
    
    # Using the sweep: sort by x. For x in (x_i, x_{i-1}), the cross-section
    # is the 2D region dominated by all points with x <= x_{i-1}... no, with x <= x.
    # Since x is between x_i and x_{i-1}, the points with x <= x are exactly
    # points 0, 1, ..., i-1 (assuming sorted ascending, x_0 < x_1 < ...).
    
    # So for x in (x_i, x_{i-1}), the 2D cross-section HV is HV_2d of
    # {(y_j, z_j) : j = 0, ..., i-1} w.r.t. (ref[1], ref[2]).
    
    # The volume contribution is (x_{i-1} - x_i) * HV_2d(S_{i-1})
    # where S_{i-1} = {points 0 to i-1}.
    
    # And for x in (ref[0], x_0), the cross-section is empty (no points), so 0.
    # Wait, no: for x in (x_0, ref[0]), the points with x <= x are {point 0}.
    # Hmm, I'm getting confused with the indexing.
    
    # Let me redefine: sort points by x ascending: x_0 <= x_1 <= ... <= x_{n-1}.
    # The intervals are: [x_{n-1}, ref[0]], [x_{n-2}, x_{n-1}], ..., [x_0, x_1], and [x_0, ref[0]]... 
    # No wait. The reference point is at ref[0] which is larger than all x_i (since points are better).
    # So the intervals from left to right are:
    # [x_0, x_1], [x_1, x_2], ..., [x_{n-2}, x_{n-1}], [x_{n-1}, ref[0]]
    
    # For x in [x_i, x_{i+1}], the points with x <= x are {0, 1, ..., i}.
    # So the 2D cross-section is HV_2d({(y_j, z_j) : j=0..i}).
    # The volume contribution is (x_{i+1} - x_i) * HV_2d(S_i) where S_i = {0..i}.
    
    # For x in [x_{n-1}, ref[0]], the points with x <= x are all points.
    # Contribution: (ref[0] - x_{n-1}) * HV_2d(S_{n-1}).
    
    # So: HV_3d = sum_{i=0}^{n-1} (x_{i+1} - x_i) * HV_2d(S_i)
    # where x_n = ref[0] and S_i = {points 0..i}.
    
    # Let me implement this:
    total_hv = 0.0
    active_2d = []  # list of (y, z) for points 0..i
    for i in range(n):
        # Add point i to active set
        active_2d.append(pts_sorted[i, 1:3])
        # Compute 2D HV of active_2d
        if len(active_2d) == 1:
            y, z = active_2d[0]
            hv2 = (ref[1] - y) * (ref[2] - z) if y < ref[1] and z < ref[2] else 0.0
        else:
            arr = np.array(active_2d)
            hv2 = _hv_2d(arr, np.array([ref[1], ref[2]]))
        
        # Width: from x_i to x_{i+1} (or ref[0] if i is last)
        if i < n - 1:
            width = pts_sorted[i + 1, 0] - pts_sorted[i, 0]
        else:
            width = ref[0] - pts_sorted[i, 0]
        
        total_hv += width * hv2
    
    return float(total_hv)


class Optimizer:
    """Structure-aware multi-objective optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.dim
        m = self.n_obj
        # DTLZ2: g depends on x[0:n-m], position on front depends on x[n-m:n]
        # For Pareto optimality, set x[0:n-m] = 0.5 (minimizes g to 0)
        # The last m variables determine position on the unit sphere
        
        n_pos = m  # number of position variables
        n_g = n - m  # number of g-variables
        
        def sample_sphere_point(theta):
            """Convert angles to point on unit sphere surface (positive orthant).
            theta: array of shape (m-1,) with values in [0, pi/2]
            Returns m objective values that sum of squares = 1.
            """
            # Use recursive sine/cosine decomposition
            # For 3D: f1 = sin(t1)*sin(t2), f2 = sin(t1)*cos(t2), f3 = cos(t1)
            # For general m:
            # f1 = prod_{i=1}^{m-1} sin(t_i)
            # f_k = sin(t_k) * prod_{i=1}^{k-1} cos(t_i) for k=2..m-1
            # f_m = prod_{i=1}^{m-1} cos(t_i)
            
            f = np.zeros(m)
            prod_sin = 1.0
            prod_cos = 1.0
            for i in range(m - 1):
                s = np.sin(theta[i])
                c = np.cos(theta[i])
                f[i] = prod_sin * s
                prod_sin *= s
                prod_cos *= c
            f[m - 1] = prod_cos
            return f
        
        def angles_to_x(theta, g_vals=None):
            """Convert angles and g-values to a full decision vector."""
            x = np.zeros(n)
            if g_vals is not None:
                x[:n_g] = g_vals
            else:
                x[:n_g] = 0.5  # Pareto-optimal g
            x[n_g:] = theta
            return x
        
        def x_to_angles(x):
            """Extract angles from a decision vector (last m variables)."""
            return x[n_g:]
        
        # Latin Hypercube Sampling for the angles
        def lhs_sample(n_samples, n_dims, rng):
            """Generate n_samples points in [0, pi/2]^n_dims using LHS."""
            points = np.zeros((n_samples, n_dims))
            for d in range(n_dims):
                # Divide [0, pi/2] into n_samples intervals
                edges = np.linspace(0, np.pi / 2, n_samples + 1)
                perm = rng.permutation(n_samples)
                for i in range(n_samples):
                    points[perm[i], d] = edges[i] + rng.random() * (edges[i + 1] - edges[i])
            return points
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Structure-aware LHS initialization
        # Use most of the budget for initialization
        init_budget = int(self.budget * 0.6)
        n_init = min(init_budget, 120)
        
        # Sample angles uniformly on the sphere surface
        # For uniform sampling on the surface of the (m-1)-dimensional simplex
        # mapped to the sphere, we can use LHS on the angles
        thetas = lhs_sample(n_init, n_pos, rng)
        
        # Create decision vectors with g = 0.5 (Pareto-optimal)
        X = np.zeros((n_init, n))
        X[:, :n_g] = 0.5
        X[:, n_g:] = thetas
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Phase 2: Adaptive mutation to fill gaps
        if used < self.budget:
            # Population size for the evolutionary phase
            pop_size = min(n_init, 80)
            
            # Keep the best pop_size points based on hypervolume contribution
            # For simplicity, keep all if n_init <= pop_size
            if n_init > pop_size:
                # Simple selection: keep points that are non-dominated
                # and spread out
                # Use a simple crowding distance or just keep all
                pass
            
            # Adaptive step size
            initial_step = 0.3
            decay = 0.995
            
            step = initial_step
            remaining = self.budget - used
            
            # We'll do random perturbations of the angles
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                
                # Pick a random parent
                parent_idx = rng.integers(0, n_init)
                parent_theta = X[parent_idx, n_g:].copy()
                
                # Mutate: add Gaussian noise to angles
                noise = step * rng.standard_normal(n_pos)
                child_theta = np.clip(parent_theta + noise, 0, np.pi / 2)
                
                # Also occasionally perturb the g-variables (to explore feasibility)
                # But for DTLZ2, g=0.5 is optimal, so keep it
                child_x = np.zeros(n)
                child_x[:n_g] = 0.5
                child_x[n_g:] = child_theta
                
                fc = problem(child_x)
                used += 1
                
                # Replace the worst point in the population
                # Use a simple criterion: maximize sum of objectives (worst is largest sum)
                # But this is biased. Better: use hypervolume contribution.
                # For speed, use a simple replacement: replace random point
                # or the point with the largest sum of objectives
                
                # Simple approach: replace the point with the largest sum
                sums = np.sum(F, axis=1)
                worst_idx = np.argmax(sums)
                
                # But we should only replace if the child is not dominated
                # by all points in the population
                # For speed, just replace the worst
                X[worst_idx] = child_x
                F[worst_idx] = fc
                
                # Decay step size
                step *= decay
            
            # Final phase: fine-tuning
            if used < self.budget:
                fine_budget = self.budget - used
                fine_step = 0.05
                for _ in range(fine_budget):
                    if time.time() > deadline:
                        break
                    
                    parent_idx = rng.integers(0, n_init)
                    parent_theta = X[parent_idx, n_g:].copy()
                    
                    noise = fine_step * rng.standard_normal(n_pos)
                    child_theta = np.clip(parent_theta + noise, 0, np.pi / 2)
                    
                    child_x = np.zeros(n)
                    child_x[:n_g] = 0.5
                    child_x[n_g:] = child_theta
                    
                    fc = problem(child_x)
                    used += 1
                    
                    sums = np.sum(F, axis=1)
                    worst_idx = np.argmax(sums)
                    
                    # Only replace if child is better in at least one objective
                    # and not worse in all
                    if np.any(fc < F[worst_idx]) and not np.all(fc > F[worst_idx]):
                        X[worst_idx] = child_x
                        F[worst_idx] = fc
        
        return X
# EVOLVE-BLOCK-END
