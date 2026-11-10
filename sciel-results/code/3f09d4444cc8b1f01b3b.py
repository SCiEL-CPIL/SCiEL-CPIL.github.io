# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = 620  # Leave 10s margin

    def get_hex_vertices(cx, cy, d, angle_deg):
        angles = np.deg2rad(angle_deg + np.arange(6) * 60)
        return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

    def hex_edge_normals(verts):
        """Return 6 unit edge normals (outward) for a CCW hexagon."""
        normals = np.zeros((6, 2))
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n = n / (np.linalg.norm(n) + 1e-12)
            normals[i] = n
        return normals

    def sat_separation(verts1, normals1, verts2, normals2):
        """
        Returns the minimum separation (positive = separated, negative = overlapping).
        Uses all 12 edge normals from both hexagons.
        """
        # Project both hexagons onto all 12 axes
        axes = np.vstack([normals1, normals2])  # (12, 2)
        proj1 = verts1 @ axes.T  # (6, 12)
        proj2 = verts2 @ axes.T  # (6, 12)
        
        min1 = proj1.min(axis=0)  # (12,)
        max1 = proj1.max(axis=0)
        min2 = proj2.min(axis=0)
        max2 = proj2.max(axis=0)
        
        # Separation on each axis: max1 - min2 or max2 - min1
        sep1 = max1 - min2  # positive means hex1 is "before" hex2
        sep2 = max2 - min1  # positive means hex2 is "before" hex1
        
        # The actual separation is the max over all axes of min(sep1, sep2)
        # If any axis has positive separation, they are separated
        # The minimum penetration depth is -min(0, min over axes of min(sep1, sep2))
        # Actually, for two convex polygons, they are separated iff there exists an axis
        # where the projections don't overlap.
        # Overlap on axis i: min1[i] <= max2[i] AND min2[i] <= max1[i]
        # Separation on axis i: max1[i] < min2[i] OR max2[i] < min1[i]
        # The "separation distance" on axis i is max(max1[i] - min2[i], max2[i] - min1[i])
        # If all are <= 0, they overlap. The minimum overlap is -min(0, min(sep1, sep2) per axis)
        
        # For each axis, the separation is:
        # if max1[i] < min2[i]: sep = max1[i] - min2[i] (negative)
        # if max2[i] < min1[i]: sep = max2[i] - min1[i] (negative)
        # else: sep = 0 (overlap or touch)
        # We want the maximum separation over all axes. If max_sep > 0, separated.
        # The penalty is max(0, -max_sep) = max(0, min over axes of ... )
        
        # Simpler: for each axis, compute the overlap amount:
        # overlap_i = max(0, min(max1[i], max2[i]) - max(min1[i], min2[i]))
        # Total overlap = max(0, max over axes of ... ) no, that's not right either.
        
        # Actually for SAT: polygons are separated if ANY axis shows separation.
        # The separation distance on axis i is:
        # d_i = max(max1[i] - min2[i], max2[i] - min1[i])
        # If d_i > 0 for any i, they are separated by at least d_i in that direction.
        # The minimum penetration (if overlapping) is:
        # min_pen = min over all i of (min1[i] + (max1[i]-min1[i])/2 + (max2[i]-min2[i])/2 ... )
        
        # Let me just use: for each axis, the gap is:
        # gap_i = max(max1[i], max2[i]) - max(min1[i], min2[i]) ... no
        
        # Standard SAT: two convex polys are separated iff there exists an axis n such that
        # max(dot(v1, n)) < min(dot(v2, n)) or max(dot(v2, n)) < min(dot(v1, n))
        # The separation distance along axis n is:
        #   if max1 < min2: sep = min2 - max1 (positive)
        #   if max2 < min1: sep = min1 - max2 (positive)
        #   else: sep = 0 (overlap on this axis)
        # The polygons are separated iff max over all axes of sep > 0.
        # The minimum translation distance is min over axes of the overlap depth when overlapping.
        
        # For penalty: we want a smooth measure of "how much they overlap"
        # Use: penalty = max(0, -min_over_axes_of_separation)
        # where separation on axis i = max(max1[i] - min2[i], max2[i] - min1[i])
        # If all separations are <= 0, they overlap. The worst-case overlap is the min separation.
        
        sep_per_axis = np.maximum(max1 - min2, max2 - min1)  # (12,)
        min_sep = sep_per_axis.min()  # most "overlapping" axis
        
        # If min_sep > 0, separated. Penalty = max(0, -min_sep)
        return max(0.0, -min_sep)

    def containment_violation(verts_inner, outer_verts):
        """
        For each vertex of inner hex, check if it's inside outer hex.
        Returns total violation (sum of squared distances for vertices outside).
        """
        # outer_verts is (6, 2), CCW
        # For each outer edge, compute cross product for all 6 inner vertices
        total_viol = 0.0
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            e = p2 - p1
            # v = inner_verts - p1, shape (6, 2)
            v = verts_inner - p1  # (6, 2)
            # cross = e[0]*v[1] - e[1]*v[0]
            cross = e[0] * v[:, 1] - e[1] * v[:, 0]  # (6,)
            # For CCW polygon, inside means cross >= 0
            # Violation: cross < 0
            viol = np.maximum(0, -cross)
            total_viol += np.sum(viol ** 2)
        return total_viol

    def compute_penalty(centers, angles, s):
        """
        centers: (12, 2)
        angles: (12,) in degrees
        s: outer hex side length
        Returns total penalty.
        """
        outer_verts = get_hex_vertices(0, 0, s, 0)
        
        # Pre-compute all inner hex vertices and normals
        inner_verts = []
        inner_normals = []
        for i in range(12):
            cx, cy = centers[i]
            ang = angles[i]
            verts = get_hex_vertices(cx, cy, 1, ang)
            inner_verts.append(verts)
            inner_normals.append(hex_edge_normals(verts))
        
        pen = 0.0
        
        # Containment: each inner hex must be inside outer
        for i in range(12):
            pen += 100.0 * containment_violation(inner_verts[i], outer_verts)
        
        # Overlap: each pair of inner hexes must be separated
        for i in range(12):
            for j in range(i + 1, 12):
                sep = sat_separation(inner_verts[i], inner_normals[i], inner_verts[j], inner_normals[j])
                pen += 100.0 * sep ** 2
        
        return pen

    def objective(vars):
        """
        vars: [s, x1, y1, a1, x2, y2, a2, ..., x12, y12, a12]
        Total: 1 + 12*3 = 37 variables
        """
        s = vars[0]
        centers = vars[1:37].reshape(12, 3)[:, :2]
        angles = vars[1:37].reshape(12, 3)[:, 2]
        
        # Objective: minimize s + penalty
        pen = compute_penalty(centers, angles, s)
        return s + pen

    def objective_grad_approx(vars, h=1e-6):
        """Simple finite difference gradient (slow but reliable)."""
        grad = np.zeros_like(vars)
        f0 = objective(vars)
        for i in range(len(vars)):
            vars_p = vars.copy()
            vars_p[i] += h
            grad[i] = (objective(vars_p) - f0) / h
        return grad

    # Initial configuration: hexagonal lattice
    # For 12 hexagons in a hexagonal arrangement:
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    # Or better: 3 rows of 4, or 4 rows of 3
    
    # Hexagonal lattice: adjacent hexagons (side 1) have center distance sqrt(3)
    # in the "flat-top" orientation (angle=0), the distance between centers
    # of touching hexagons is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)
    
    # Let's use a 3x4 grid with hexagonal offset
    # Actually, for flat-top hexagons (angle=0):
    # Horizontal spacing between centers: 3 (for side 1, width = 2, so center-to-center = 3 for no overlap? No)
    # For flat-top hex with side 1:
    # Width (horizontal) = 2 * side = 2
    # Height (vertical) = sqrt(3) * side = sqrt(3)
    # To pack without overlap, horizontal center distance >= 3 (width + gap), vertical >= sqrt(3)
    # But with hexagonal offset, we can do better.
    
    # Actually for two flat-top unit hexagons side by side:
    # The minimum center distance is sqrt(3) in the vertical direction and 3 in horizontal?
    # No. Let me think more carefully.
    
    # A flat-top regular hexagon with side 1 has:
    # - Flat top and bottom edges of length 1
    # - Pointed left and right vertices
    # - Width (left to right) = 2
    # - Height (top to bottom) = sqrt(3)
    
    # Two such hexagons placed side by side horizontally:
    # Minimum horizontal center distance = 2 (they touch at the pointed vertices)
    # But that's not right either. Let me use the SAT check.
    
    # For now, use a reasonable initial guess with some spacing.
    # Hexagonal lattice: centers at (i * 3 + (j % 2) * 1.5, j * sqrt(3)/2 * 2)
    # Wait, for flat-top hexagons, the lattice vectors are:
    # a1 = (3, 0) -- no, that's too much
    # Actually, for two flat-top unit hexagons to just touch:
    # If placed with centers at distance d apart horizontally, they touch when d = 2 (the width).
    # But the hexagons also have slanted edges.
    
    # Let me just use a grid with generous spacing and let the optimizer tighten it.
    # 4 columns, 3 rows
    # x spacing: 3.0 (generous)
    # y spacing: 2.5 (generous)
    
    init_centers = np.zeros((12, 2))
    init_angles = np.zeros(12)
    idx = 0
    for row in range(3):
        for col in range(4):
            x = (col - 1.5) * 3.0
            y = (row - 1) * 2.5
            init_centers[idx] = [x, y]
            init_angles[idx] = 0.0
            idx += 1
    
    # Estimate initial s
    # The outer hex needs to contain all these hexagons.
    # Max |x| = 1.5 * 3 + 1 = 5.5, max |y| = 1 * 2.5 + sqrt(3)/2 ≈ 3.37
    # For a regular hexagon with side s, the max distance from center to a vertex is s.
    # The max distance from center to an edge (apothem) is s * sqrt(3)/2.
    # We need s * sqrt(3)/2 >= max distance from origin to any point of any inner hex.
    # Rough estimate: s >= 4.5
    init_s = 5.0
    
    # Build initial vars
    init_vars = np.zeros(37)
    init_vars[0] = init_s
    for i in range(12):
        init_vars[1 + 3*i] = init_centers[i, 0]
        init_vars[2 + 3*i] = init_centers[i, 1]
        init_vars[3 + 3*i] = init_angles[i]
    
    # Optimize
    best_vars = init_vars.copy()
    best_obj = objective(init_vars)
    
    # Try multiple optimization methods and restarts
    methods_to_try = [
        ('L-BFGS-B', {'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}),
        ('Nelder-Mead', {'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}),
    ]
    
    restarts = 5
    np.random.seed(42)
    
    for restart in range(restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x0 = init_vars.copy()
        else:
            # Perturb
            x0 = init_vars.copy()
            x0[1:] += np.random.normal(0, 0.3, 36)
            x0[0] = init_s + np.random.uniform(-0.5, 0.5)
        
        for method_name, options in methods_to_try:
            if time.time() > deadline:
                break
            
            try:
                res = minimize(objective, x0, method=method_name, options=options)
                obj_val = res.fun
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_vars = res.x.copy()
                if time.time() > deadline:
                    break
            except Exception:
                pass
        
        # Also try a simple gradient descent with line search
        if time.time() > deadline:
            break
        
        # Custom gradient descent
        x_cur = best_vars.copy()
        lr = 0.05
        for step in range(200):
            if time.time() > deadline:
                break
            # Compute gradient via finite differences (expensive but robust)
            grad = np.zeros_like(x_cur)
            h = 1e-5
            f0 = objective(x_cur)
            for i in range(len(x_cur)):
                x_p = x_cur.copy()
                x_p[i] += h
                grad[i] = (objective(x_p) - f0) / h
            
            # Update
            x_new = x_cur - lr * grad
            # Ensure s >= sqrt(12)
            x_new[0] = max(x_new[0], np.sqrt(12) + 1e-6)
            
            f_new = objective(x_new)
            if f_new < f0:
                x_cur = x_new
                if f_new < best_obj:
                    best_obj = f_new
                    best_vars = x_cur.copy()
                # Decrease learning rate
                lr *= 0.98
            else:
                lr *= 0.5
                if lr < 1e-8:
                    break
    
    # Extract result
    s_best = best_vars[0]
    centers = best_vars[1:37].reshape(12, 3)[:, :2]
    angles = best_vars[1:37].reshape(12, 3)[:, 2]
    
    # Validate and fix if needed
    # Ensure s >= sqrt(12)
    s_best = max(s_best, np.sqrt(12) + 1e-9)
    
    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = angles[i]
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_best

# EVOLVE-BLOCK-END
