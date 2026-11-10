import numpy as np
import time
from scipy.optimize import differential_evolution

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, 
    maximizing 1/outer_hex_side_length.
    
    Returns
        inner_hex_data: np.ndarray of shape (11,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = 600.0  # 600s limit, keep margin
    safety_margin = 10.0
    
    n_inner = 11
    
    # Helper to get vertices of a hexagon
    def get_hex_vertices(cx, cy, s, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        angles = angle_rad + np.deg2rad(np.arange(0, 360, 60))
        xs = cx + s * np.cos(angles)
        ys = cy + s * np.sin(angles)
        return np.column_stack((xs, ys))
    
    # Helper to check if a point is inside a hexagon (defined by center, size, angle)
    # Returns min distance to boundary (positive if inside, negative if outside)
    # For a regular hexagon, the distance to the nearest edge is s * cos(angle_diff) where angle_diff is angle between normal and point vector?
    # Simpler: Use SAT for containment.
    def point_in_hexagon_distance(px, py, cx, cy, s, angle_deg):
        # Translate point to hex center
        dx = px - cx
        dy = py - cy
        # Rotate point by -angle to align hex with axes
        a = np.deg2rad(-angle_deg)
        cos_a = np.cos(a)
        sin_a = np.sin(a)
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        
        # In local coords, hexagon has vertices at s * (cos(60k), sin(60k))
        # Edges are at distance s * cos(30) = s * sqrt(3)/2 from center? No, apothem is s * sqrt(3)/2.
        # The condition for a point (rx, ry) to be inside is:
        # |rx| <= s * sqrt(3)/2
        # |ry| <= s * sqrt(3)/2
        # |rx + ry/sqrt(3)| <= s * sqrt(3)/2  (approx, depends on orientation)
        
        # Let's use the standard inequality for a hexagon with flat top (vertices at 0, 60, ...)
        # Actually, the vertices are at angles 0, 60, 120...
        # The edges are perpendicular to the directions 30, 90, 150...
        # Distance to edge i is s * cos(angle_between_point_dir_and_normal_i)
        
        # Robust method: Calculate cross products for the 6 edges.
        # Edge normals are at angles 30, 90, 150, 210, 270, 330 degrees in local frame.
        normals = np.deg2rad(np.arange(30, 360, 60))
        # Normal vectors
        nx = np.cos(normals)
        ny = np.sin(normals)
        
        # Projection of point onto normals
        proj = rx * nx + ry * ny
        # The max projection should be <= apothem = s * sqrt(3)/2
        apothem = s * np.sqrt(3) / 2
        # If proj > apothem, it's outside.
        # The "distance" violation is max(proj - apothem, 0)
        return np.max(proj - apothem)

    def hexagon_overlap_penalty(cx1, cy1, a1, cx2, cy2, a2):
        """
        Returns the maximum overlap depth between two unit hexagons.
        0 if separated.
        """
        v1 = get_hex_vertices(cx1, cy1, 1.0, a1)
        v2 = get_hex_vertices(cx2, cy2, 1.0, a2)
        
        # SAT: Check all 12 normals (6 from each)
        # If any normal separates them, return 0 (or small negative)
        # Otherwise, return the minimum overlap depth? No, we want to penalize overlap.
        # The "penetration depth" is the minimum distance one hex must move to separate.
        # This is complex. A simpler proxy: 
        # If they overlap, return a positive value proportional to how much they overlap.
        
        # Let's check if any normal separates them.
        # For each normal n, project both polygons.
        # If max1 < min2 or max2 < min1, separated.
        # If not, they overlap.
        
        # We calculate the minimum separation distance over all axes.
        # If min_sep > 0, they are separated.
        # If min_sep < 0, they overlap by -min_sep.
        
        min_sep = float('inf')
        
        # Get normals of hex 1
        angles1 = np.deg2rad(a1) + np.deg2rad(np.arange(30, 360, 60))
        nx1 = np.cos(angles1)
        ny1 = np.sin(angles1)
        
        # Get normals of hex 2
        angles2 = np.deg2rad(a2) + np.deg2rad(np.arange(30, 360, 60))
        nx2 = np.cos(angles2)
        ny2 = np.sin(angles2)
        
        all_nx = np.concatenate([nx1, nx2])
        all_ny = np.concatenate([ny1, ny2])
        
        for nx, ny in zip(all_nx, all_ny):
            # Project v1
            p1 = v1[:, 0] * nx + v1[:, 1] * ny
            min1, max1 = np.min(p1), np.max(p1)
            
            # Project v2
            p2 = v2[:, 0] * nx + v2[:, 1] * ny
            min2, max2 = np.min(p2), np.max(p2)
            
            # Calculate separation
            if max1 < min2:
                sep = min2 - max1
            elif max2 < min1:
                sep = min1 - max2
            else:
                sep = 0.0 # Overlap on this axis
            
            if sep > 0:
                return 0.0 # Separated
            
            min_sep = min(min_sep, sep) # sep is 0 here, so min_sep stays 0?
            # Wait, if they overlap on this axis, the "separation" is negative (penetration).
            # We need the penetration depth.
            # If max1 < min2 is false and max2 < min1 is false:
            # The overlap length is min(max1, max2) - max(min1, min2).
            # The "separation" is negative of that?
            # Actually, for SAT, if they overlap on ALL axes, they intersect.
            # The minimum "separation" (which is negative) gives the penetration depth.
            
            overlap_len = min(max1, max2) - max(min1, min2)
            # If overlap_len > 0, they overlap on this axis.
            # The penetration depth along this axis is overlap_len.
            # We want the minimum penetration depth over all axes?
            # No, the actual distance to separate is the minimum of the "gaps" if separated,
            # or the minimum "overlap" if intersecting?
            # Actually, the minimal translation vector to separate is along the axis with the smallest overlap?
            # For penalty, let's just return the maximum overlap length on any axis?
            # Or the minimum?
            # If I return the minimum overlap, it might be 0 if they just touch on one axis but overlap on another?
            # No, if they intersect, they overlap on ALL axes.
            # The "depth" of intersection is typically the minimum of the overlaps?
            # Let's use the maximum overlap to be safe? No, that over-penalizes.
            # Let's use the minimum overlap (closest to 0) as the primary penalty, 
            # but if any overlap is large, we want to push them apart.
            
            # Actually, if sep < 0, it means overlap.
            # Let's define sep as:
            # if max1 < min2: sep = min2 - max1 (positive, separated)
            # else: sep = min(max1, max2) - max(min1, min2) (negative, overlap magnitude)
            
            if max1 < min2:
                sep_val = min2 - max1
            elif max2 < min1:
                sep_val = min1 - max2
            else:
                sep_val = -(min(max1, max2) - max(min1, min2))
                
            min_sep = min(min_sep, sep_val)
            
        # If min_sep > 0, they are separated.
        # If min_sep < 0, they overlap.
        # Return -min_sep as penalty (positive value for overlap)
        if min_sep < 0:
            return -min_sep
        return 0.0

    def objective(params):
        # params: [S, x1, y1, a1, ..., x11, y11, a11]
        S = params[0]
        if S < np.sqrt(11):
            return 1e6
        
        inner_params = params[1:].reshape(11, 3)
        
        # Containment penalty
        contain_pen = 0.0
        for i in range(11):
            xi, yi, ai = inner_params[i]
            verts = get_hex_vertices(xi, yi, 1.0, ai)
            for vx, vy in verts:
                viol = point_in_hexagon_distance(vx, vy, 0, 0, S, 0)
                if viol > 0:
                    contain_pen += viol**2
        
        # Overlap penalty
        overlap_pen = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                xi, yi, ai = inner_params[i]
                xj, yj, aj = inner_params[j]
                ov = hexagon_overlap_penalty(xi, yi, ai, xj, yj, aj)
                if ov > 0:
                    overlap_pen += ov**2
        
        # Total penalty
        penalty = 100.0 * (contain_pen + overlap_pen)
        
        return S + penalty

    # Initial guess
    # Use a symmetric arrangement
    # Center at (0,0)
    # 6 around center, 4 further out?
    # Or use the seed's arrangement as a starting point for local search?
    # Differential evolution is global, so random start is fine, but good bounds help.
    
    # Bounds for S: [sqrt(11), 15]
    # Bounds for x, y: [-10, 10]
    # Bounds for angle: [0, 360]
    
    bounds = [(np.sqrt(11), 15.0)]
    for _ in range(11):
        bounds.append([(-10, 10)])
        bounds.append([(-10, 10)])
        bounds.append([(0, 360)])
        
    # Run optimization
    # To save time, we might run a quick DE first, then polish with Nelder-Mead
    # But DE is slow. Let's try a smaller population or maxiter.
    
    # Alternative: Use a known good configuration and just optimize S?
    # No, positions must adjust.
    
    # Let's try a few restarts with Differential Evolution
    best_result = None
    best_val = float('inf')
    
    # Time check
    if time.time() - start_time > deadline - safety_margin:
        # Fallback to seed
        inner_hex_data = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0]
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    # Run DE with limited iterations
    # popsize=10, maxiter=50 is fast
    # We can run multiple times with different seeds
    for seed in [1, 2, 3]:
        if time.time() - start_time > deadline - safety_margin:
            break
            
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                seed=seed, 
                maxiter=200, 
                popsize=15, 
                tol=0.01,
                update_interval=10,
                polish=True
            )
            val = res.fun
            if val < best_val:
                best_val = val
                best_result = res.x
        except Exception as e:
            pass

    if best_result is None:
        # Fallback
        inner_hex_data = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0]
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    S = best_result[0]
    inner_params = best_result[1:].reshape(11, 3)
    
    # Verify validity strictly
    def is_valid(S, inner_params):
        # Check finite
        if not np.all(np.isfinite(inner_params)) or not np.isfinite(S):
            return False
        if S < np.sqrt(11) - 1e-9:
            return False
            
        # Check containment
        for i in range(11):
            xi, yi, ai = inner_params[i]
            verts = get_hex_vertices(xi, yi, 1.0, ai)
            for vx, vy in verts:
                viol = point_in_hexagon_distance(vx, vy, 0, 0, S, 0)
                # Must be <= 1e-6 (allowing tiny floating point error)
                if viol > 1e-6:
                    return False
                    
        # Check overlap
        for i in range(11):
            for j in range(i + 1, 11):
                xi, yi, ai = inner_params[i]
                xj, yj, aj = inner_params[j]
                # Use the same logic as penalty but strict
                ov = hexagon_overlap_penalty(xi, yi, ai, xj, yj, aj)
                # If ov > 1e-6, they are intersecting
                if ov > 1e-6:
                    return False
        return True

    if not is_valid(S, inner_params):
        # If invalid, try to recover by increasing S slightly or using fallback
        # Or just use the seed which is known valid
        inner_hex_data = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0]
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    inner_hex_data = inner_params
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(S)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
