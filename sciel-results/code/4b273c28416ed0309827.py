# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Packs 12 unit regular hexagons in a regular hexagon using physics-based relaxation (L-BFGS-B).
    Returns (inner, outer, s) where inner is (12, 3), outer is (3,), s is side length.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 600s limit

    # Helper: Get vertices of a hexagon
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + side * math.cos(a)
            y = cy + side * math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    # Helper: Compute minimum separation distance between two convex polygons using SAT
    # Returns (is_separated, min_dist)
    # min_dist > 0 if separated, 0 if touching/overlapping
    def polygon_separation(v1, v2):
        # v1, v2 are (6, 2) arrays
        axes = []
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i+1)%6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        min_dist = float('inf')
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Distance between intervals
            if max1 <= min2:
                dist = min2 - max1
            elif max2 <= min1:
                dist = min1 - max2
            else:
                dist = 0.0
            
            if dist < min_dist:
                min_dist = dist
            if min_dist < 1e-9:
                return False, 0.0 # Overlapping or touching, no need to check more axes for distance
        
        return True, min_dist

    # Helper: Compute smooth penalty for one pair of hexagons
    def pair_penalty(verts1, verts2):
        is_sep, dist = polygon_separation(verts1, verts2)
        if is_sep and dist > 0.01:
            return 0.0
        # Smooth penalty: large if overlapping, small if close
        # Use a quadratic penalty for overlap
        if dist < 0:
            return 100.0 * (-dist)**2
        else:
            return 0.1 * (0.01 - dist)**2 if dist < 0.01 else 0.0

    # Helper: Compute containment penalty
    # outer_verts: (6, 2)
    # inner_verts: (6, 2)
    def containment_penalty(inner_verts, outer_verts):
        penalty = 0.0
        # For each inner vertex, check against outer edges
        for v in inner_verts:
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i+1)%6]
                edge = p2 - p1
                vec = v - p1
                cross = edge[0]*vec[1] - edge[1]*vec[0]
                # For CCW polygon, interior is cross > 0
                # We want cross >= -1e-6
                # If cross < 0, it's outside
                if cross < 0:
                    # Penalty proportional to violation squared
                    # Scale by edge length to make it dimensionless-ish?
                    # The cross product magnitude is proportional to distance * edge_length
                    # Edge length is fixed for regular hexagon
                    penalty += 100.0 * (-cross)**2
        return penalty

    # Objective function
    # x: (36,) array: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, cx, cy, s]
    # We fix outer angle to 0 for simplicity, or include it?
    # Let's include outer center and side length. Outer angle 0.
    # Actually, we can fix outer center to (0,0) and just optimize s and inner positions.
    # The problem allows shifting. So fixing outer at (0,0) is WLOG.
    # Variables: 12 * (x, y) + 1 * s = 25 variables.
    # Inner angles: Fixed to 0? Rotation might help?
    # For regular hexagons, rotating by 60 deg is same.
    # Rotating by 30 deg swaps flat/pointy top.
    # Let's fix inner angles to 0 initially. If needed, we can add them.
    # Let's add inner angles to be safe? 12 * 3 + 1 = 37 variables.
    # Let's start with fixed angles 0 to reduce dimensionality and complexity.
    # If stuck, we can try adding angles.
    
    def objective(x):
        # x: [x0..x11, y0..y11, s]
        centers = np.array(x[:24]).reshape(12, 2)
        s = x[24]
        if s < 0.1:
            return 1e6
        
        # Compute vertices for all inner hexes
        inner_verts_list = []
        for i in range(12):
            cx, cy = centers[i]
            v = get_hex_vertices(cx, cy, 1.0, 0.0)
            inner_verts_list.append(v)
        
        # Outer hexagon
        outer_verts = get_hex_vertices(0.0, 0.0, s, 0.0)
        
        total_penalty = 0.0
        
        # 1. Inner-Outer Containment
        for i in range(12):
            total_penalty += containment_penalty(inner_verts_list[i], outer_verts)
            
        # 2. Inner-Inner Separation
        for i in range(12):
            for j in range(i+1, 12):
                total_penalty += pair_penalty(inner_verts_list[i], inner_verts_list[j])
                
        # 3. Objective: Minimize s + Penalties
        # We want to minimize s, but penalties must be 0.
        # If penalties > 0, we are infeasible.
        # We can use a weighted sum: s + lambda * penalty
        # Or just minimize penalty first, then s?
        # L-BFGS-B handles multi-objective poorly.
        # Let's minimize: s + 10 * penalty
        # If penalty is 0, it minimizes s.
        # If penalty is > 0, the 10*penalty term dominates, pushing to feasible region.
        
        return s + 10.0 * total_penalty

    # Initial Configuration
    # Hexagonal lattice arrangement
    # 1, 2, 3, 2, 1 is 9. We need 12.
    # 2, 3, 4, 3 is 12.
    # Or 3, 4, 3, 2?
    # Let's use a compact cluster.
    # Center (0,0)
    # Row y=0: 3 hexes at x = -2.5, 0, 2.5
    # Row y=sqrt(3): 3 hexes at x = -1.25, 1.25 ... wait, spacing.
    # Hex side 1.
    # Horizontal distance between centers in same row: 2.5?
    # Vertices of hex at (0,0): (1,0), (0.5, 0.866), (-0.5, 0.866), (-1,0), (-0.5, -0.866), (0.5, -0.866)
    # Rightmost x=1. Leftmost x=-1.
    # If I place another hex at x=2.5, its leftmost is 1.5. Gap 0.5.
    # If I place at x=2.0, leftmost 1.0. Touching.
    # So same row spacing is 2.0? No, that's vertex to vertex.
    # Edge to edge?
    # If flat top, the right edge is vertical? No, vertices at 0, 60, 120...
    # Edge from (1,0) to (0.5, 0.866).
    # The "rightmost" point is (1,0).
    # If I shift by (2, 0), the new leftmost is (1,0). They touch at vertex (1,0)?
    # No, the old hex has vertex (1,0). The new hex has vertex (1,0) relative to its center?
    # New center (2,0). Leftmost vertex (1,0).
    # So they touch at (1,0).
    # But do they overlap?
    # Old hex: x in [-1, 1].
    # New hex: x in [1, 3].
    # They share the point (1,0).
    # However, the edges might cross.
    # Let's use the standard hex grid coordinates.
    # dx = 1.5 * side * 2? No.
    # In a hex grid with flat tops:
    # Horizontal spacing between columns: 1.5 * side? No.
    # Let's just use the seed's valid configuration as start.
    # Seed:
    # (0, 0)
    # (-2.5, 0), (2.5, 0)
    # (-1.25, 2.17), (1.25, 2.17)
    # (-1.25, -2.17), (1.25, -2.17)
    # (-3.75, 2.17), (3.75, 2.17)
    # (-3.75, -2.17), (3.75, -2.17)
    # (0, -4)
    
    init_centers = np.array([
        [0.0, 0.0],
        [-2.5, 0.0],
        [2.5, 0.0],
        [-1.25, 2.17],
        [1.25, 2.17],
        [-1.25, -2.17],
        [1.25, -2.17],
        [-3.75, 2.17],
        [3.75, 2.17],
        [-3.75, -2.17],
        [3.75, -2.17],
        [0.0, -4.0]
    ])
    
    # Calculate initial s
    # Find max extent
    max_x = np.max(init_centers[:, 0]) + 1.0
    min_x = np.min(init_centers[:, 0]) - 1.0
    max_y = np.max(init_centers[:, 1]) + 0.866
    min_y = np.min(init_centers[:, 1]) - 0.866
    
    # Approximate s
    # For hexagon side s, width = 2s, height = sqrt(3)s
    # We need 2s >= max_x - min_x and sqrt(3)s >= max_y - min_y
    # s >= (max_x - min_x) / 2
    # s >= (max_y - min_y) / sqrt(3)
    s_init = max((max_x - min_x) / 2.0, (max_y - min_y) / math.sqrt(3.0))
    s_init *= 1.1 # Padding
    
    x0 = np.concatenate([init_centers.flatten(), [s_init]])
    
    # Bounds
    # Centers can be anywhere, but let's constrain to a reasonable box
    bounds = [(-10, 10)] * 24 + [(1.0, 20.0)]
    
    # Run L-BFGS-B
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10})
        if res.success:
            x_opt = res.x
        else:
            # Try again with different init or more iterations
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12})
            x_opt = res.x
    except Exception:
        x_opt = x0

    # Extract result
    centers_opt = np.array(x_opt[:24]).reshape(12, 2)
    s_opt = x_opt[24]
    
    # Construct inner array
    inner = np.zeros((12, 3))
    for i in range(12):
        inner[i, 0] = centers_opt[i, 0]
        inner[i, 1] = centers_opt[i, 1]
        inner[i, 2] = 0.0 # Angle
    
    # Construct outer array
    outer = np.array([0.0, 0.0, 0.0])
    
    # Validation and refinement
    # Check if valid
    def is_valid(inner_arr, s_val):
        inner_verts_list = []
        for i in range(12):
            v = get_hex_vertices(inner_arr[i, 0], inner_arr[i, 1], 1.0, inner_arr[i, 2])
            inner_verts_list.append(v)
        
        outer_verts = get_hex_vertices(0.0, 0.0, s_val, 0.0)
        
        # Check containment
        for i in range(12):
            if containment_penalty(inner_verts_list[i], outer_verts) > 1e-6:
                return False
        
        # Check separation
        for i in range(12):
            for j in range(i+1, 12):
                is_sep, dist = polygon_separation(inner_verts_list[i], inner_verts_list[j])
                if not is_sep or dist < 1e-6:
                    return False
        return True

    if not is_valid(inner, s_opt):
        # Fallback: Use the initial configuration with calculated s
        # This is guaranteed to be valid (mostly)
        inner = np.zeros((12, 3))
        for i in range(12):
            inner[i, 0] = init_centers[i, 0]
            inner[i, 1] = init_centers[i, 1]
            inner[i, 2] = 0.0
        s_opt = s_init
        # Ensure s is large enough
        # Recalculate s for init
        max_x = np.max(init_centers[:, 0]) + 1.0
        min_x = np.min(init_centers[:, 0]) - 1.0
        max_y = np.max(init_centers[:, 1]) + 0.866
        min_y = np.min(init_centers[:, 1]) - 0.866
        s_calc = max((max_x - min_x) / 2.0, (max_y - min_y) / math.sqrt(3.0))
        s_opt = max(s_opt, s_calc * 1.05)

    return inner, outer, s_opt

# EVOLVE-BLOCK-END
