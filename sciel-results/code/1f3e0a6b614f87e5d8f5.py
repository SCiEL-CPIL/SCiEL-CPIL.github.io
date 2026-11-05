"""Hexagon packing: 11 unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize_scalar, minimize

N = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Base vertices of unit hexagon (flat-top, angle 0)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    base_vertices = np.array([
        [math.cos(math.radians(60 * k)), math.sin(math.radians(60 * k))] for k in range(6)
    ])

    # Initial "flower" arrangement:
    # 1 center + 6 inner ring (distance sqrt(3)) + 4 outer ring (distance 3.0)
    # All inner hexagons have rotation 0.
    centers = np.array([[0.0, 0.0]])
    angles = np.zeros(1)

    # Inner ring: 6 hexagons at distance sqrt(3)
    r1 = math.sqrt(3)
    for k in range(6):
        angle = math.radians(30 + k * 60)
        centers = np.vstack([centers, [r1 * math.cos(angle), r1 * math.sin(angle)]])
        angles = np.append(angles, 0.0)

    # Outer ring: 4 hexagons at distance 3.0
    r2 = 3.0
    for k in range(4):
        angle = math.radians(k * 60)
        centers = np.vstack([centers, [r2 * math.cos(angle), r2 * math.sin(angle)]])
        angles = np.append(angles, 0.0)

    n_hex = len(centers)

    def get_hex_vertices(cx, cy, angle_rad):
        """Get vertices of a hexagon with center (cx, cy) and rotation angle_rad."""
        theta = angle_rad
        verts = []
        for k in range(6):
            a = theta + k * math.pi / 3.0
            verts.append([cx + math.cos(a), cy + math.sin(a)])
        return np.array(verts)

    def min_distance_between_hexes(c1, a1, c2, a2):
        """
        Compute minimum distance between two unit hexagons.
        If negative, they overlap.
        Uses support function approach for convex polygons.
        """
        v1 = get_hex_vertices(c1[0], c1[1], a1)
        v2 = get_hex_vertices(c2[0], c2[1], a2)
        
        # Use scipy's distance between convex sets
        # For two convex polygons, we can use the Gilbert algorithm or
        # simply check all edge-edge and vertex-edge distances
        # For simplicity and speed, use a sampling approach with refinement
        
        # Actually, for hexagons, the minimum distance can be found by
        # checking all pairs of edges and vertices
        # But a faster approach: use the fact that for convex polygons,
        # min distance = max over directions of (support1 - support2)
        
        # Let's use a simpler approach: sample many directions
        # and compute the support function difference
        # This is approximate but fast
        
        n_dirs = 72
        min_dist = float('inf')
        
        for d in range(n_dirs):
            theta = 2 * math.pi * d / n_dirs
            dx = math.cos(theta)
            dy = math.sin(theta)
            
            # Support function of hex1 in direction (dx, dy)
            s1 = np.max(v1 @ np.array([dx, dy]))
            # Support function of hex2 in opposite direction
            s2 = np.min(v2 @ np.array([dx, dy]))
            
            # Distance in this direction
            dist = s1 - s2
            min_dist = min(min_dist, dist)
        
        return min_dist

    def compute_outer_side(centers_arr, angles_arr, theta_out):
        """
        Compute minimum outer hexagon side length for given inner hexagon positions
        and outer hexagon rotation.
        """
        # Outer hexagon face normals at angles 30 + k*60 + theta_out
        normals = np.array([
            [math.cos(math.radians(30 + k * 60) + theta_out),
             math.sin(math.radians(30 + k * 60) + theta_out)]
            for k in range(6)
        ])

        max_proj = 0.0
        for i in range(n_hex):
            verts = get_hex_vertices(centers_arr[i][0], centers_arr[i][1], angles_arr[i])
            dots = verts @ normals.T
            max_proj = max(max_proj, float(np.max(np.abs(dots))))

        # Apothem = S * sqrt(3)/2, so S = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / math.sqrt(3)

    def objective(params):
        """
        Objective function for optimization.
        params: [cx1, cy1, a1, cx2, cy2, a2, ..., theta_out]
        Returns: outer side length + penalty for overlaps
        """
        if time.time() > deadline - 5:
            return 1e6
        
        centers_opt = params[:3*n_hex].reshape(n_hex, 3)
        theta_out = params[3*n_hex]
        
        # Compute outer side length
        side = compute_outer_side(centers_opt[:, :2], centers_opt[:, 2], theta_out)
        
        # Check for overlaps - use pairwise distance
        # Only check nearby pairs to save time
        penalty = 0.0
        min_allowed = 1e-6  # Small tolerance
        
        for i in range(n_hex):
            for j in range(i+1, n_hex):
                # Quick bounding box check
                ci = centers_opt[i]
                cj = centers_opt[j]
                dist_center = np.linalg.norm(ci[:2] - cj[:2])
                if dist_center > 2.5:  # Hexagons can't be farther than ~2.5 and overlap
                    continue
                
                d = min_distance_between_hexes(ci[:2], ci[2], cj[:2], cj[2])
                if d < min_allowed:
                    # Penalty proportional to overlap
                    penalty += (min_allowed - d) ** 2
        
        return side + 100.0 * penalty

    # First: optimize only the outer rotation angle (fast)
    def compute_side_theta(theta):
        return compute_outer_side(centers, angles, theta)

    best_score = float('inf')
    best_theta = 0.0

    # Coarse grid search over [0, pi/3] (periodic)
    thetas = np.linspace(0, math.pi / 3, 3000)
    for t in thetas:
        if time.time() > deadline - 20:
            break
        s = compute_side_theta(float(t))
        if s < best_score:
            best_score = s
            best_theta = float(t)

    # Refine with minimize_scalar
    step = (math.pi / 3) / 3000
    lo = max(0.0, best_theta - 10 * step)
    hi = min(math.pi / 3, best_theta + 10 * step)

    try:
        res = minimize_scalar(compute_side_theta, bounds=(lo, hi), method='bounded',
                              options={'xatol': 1e-14, 'maxiter': 2000})
        if res.fun < best_score:
            best_score = res.fun
            best_theta = res.x
    except Exception:
        pass

    # Now try to optimize inner hexagon positions to potentially reduce side length
    # Use the current best configuration as starting point
    initial_params = np.zeros(3 * n_hex + 1)
    for i in range(n_hex):
        initial_params[3*i] = centers[i][0]
        initial_params[3*i+1] = centers[i][1]
        initial_params[3*i+2] = angles[i]
    initial_params[3*n_hex] = best_theta

    # Try a few local optimizations with different starting points
    best_params = initial_params.copy()
    best_obj = compute_outer_side(centers, angles, best_theta)

    # Try perturbing the initial configuration slightly
    n_attempts = 5
    for attempt in range(n_attempts):
        if time.time() > deadline - 10:
            break
        
        # Create a perturbed starting point
        if attempt == 0:
            params0 = initial_params.copy()
        else:
            params0 = initial_params.copy()
            # Add small random perturbations
            noise = np.random.randn(3 * n_hex + 1) * 0.01
            noise[3*n_hex] = 0  # Don't perturb the outer angle much
            params0 += noise
        
        try:
            result = minimize(objective, params0, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass

    # Extract the best configuration
    best_centers = best_params[:3*n_hex].reshape(n_hex, 3)
    best_theta_final = best_params[3*n_hex]
    
    # Recompute the actual side length (without penalty)
    final_side = compute_outer_side(best_centers[:, :2], best_centers[:, 2], best_theta_final)
    
    # Verify no overlaps
    valid = True
    for i in range(n_hex):
        for j in range(i+1, n_hex):
            d = min_distance_between_hexes(best_centers[i][:2], best_centers[i][2], 
                                           best_centers[j][:2], best_centers[j][2])
            if d < -1e-6:  # Significant overlap
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fall back to the original flower configuration
        final_side = best_score
        best_theta_final = best_theta
        best_centers = np.column_stack([centers, angles])

    # Final polish: optimize only theta_out with the fixed inner configuration
    if time.time() < deadline - 5:
        def side_only_theta(t):
            return compute_outer_side(best_centers[:, :2], best_centers[:, 2], t)
        
        lo2 = max(0.0, best_theta_final - 1e-8)
        hi2 = min(math.pi / 3, best_theta_final + 1e-8)
        try:
            res = minimize_scalar(side_only_theta, bounds=(lo2, hi2), method='bounded',
                                  options={'xatol': 1e-16, 'maxiter': 500})
            if res.fun < final_side:
                final_side = res.fun
                best_theta_final = res.x
        except Exception:
            pass

    # Build output
    inner = []
    for i in range(n_hex):
        inner.append([best_centers[i][0], best_centers[i][1], math.degrees(best_centers[i][2])])

    side = final_side
    theta_out_deg = math.degrees(best_theta_final)

    return inner, (0.0, 0.0), side, theta_out_deg

# EVOLVE-BLOCK-END
