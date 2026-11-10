# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = 620  # Leave 10s margin

    def get_hex_vertices(cx, cy, d, angle_deg):
        angles = np.deg2rad(angle_deg + np.arange(6) * 60)
        return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

    def point_in_hexagon(p, hex_verts):
        # hex_verts is (6,2)
        # Use cross product test
        # For a convex polygon, p is inside if it is on the same side of all edges
        inside = True
        for i in range(6):
            p1 = hex_verts[i]
            p2 = hex_verts[(i + 1) % 6]
            # Vector p2-p1
            e = p2 - p1
            # Vector p-p1
            v = p - p1
            # Cross product z-component
            cross = e[0] * v[1] - e[1] * v[0]
            # For a CCW polygon, cross should be >= 0
            # Our vertices are generated CCW?
            # cos(0), cos(60)... yes, CCW.
            if cross < -1e-8:
                inside = False
                break
        return inside

    def containment_penalty(inner_verts, outer_verts):
        pen = 0.0
        for v in inner_verts:
            if not point_in_hexagon(v, outer_verts):
                # Calculate distance to nearest edge
                min_dist = np.inf
                for i in range(6):
                    p1 = outer_verts[i]
                    p2 = outer_verts[(i + 1) % 6]
                    # Project v onto line p1-p2
                    dp = p2 - p1
                    dv = v - p1
                    t = np.dot(dv, dp) / np.dot(dp, dp)
                    t = np.clip(t, 0, 1)
                    proj = p1 + t * dp
                    dist = np.linalg.norm(v - proj)
                    if dist < min_dist:
                        min_dist = dist
                pen += min_dist * min_dist
        return pen

    def sat_overlap_penalty(hex1_verts, hex2_verts):
        # Check if two convex polygons overlap
        # If they overlap, return a penalty
        # If they don't, return 0
        
        def project(verts, axis):
            return np.dot(verts, axis)
        
        axes = []
        for i in range(6):
            p1 = hex1_verts[i]
            p2 = hex1_verts[(i + 1) % 6]
            edge = p2 - p1
            # Normal
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            axes.append(normal)
            
        for i in range(6):
            p1 = hex2_verts[i]
            p2 = hex2_verts[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            axes.append(normal)
            
        for axis in axes:
            proj1 = project(hex1_verts, axis)
            proj2 = project(hex2_verts, axis)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for separation
            if max1 < min2 or max2 < min1:
                return 0.0  # Separated on this axis, so no overlap
        
        # If we get here, they overlap on all axes
        # Calculate penalty: sum of squared distances of vertices of 1 inside 2
        pen = 0.0
        for v in hex1_verts:
            if point_in_hexagon(v, hex2_verts):
                # Find distance to nearest edge of hex2
                min_dist = np.inf
                for i in range(6):
                    p1 = hex2_verts[i]
                    p2 = hex2_verts[(i + 1) % 6]
                    dp = p2 - p1
                    dv = v - p1
                    t = np.dot(dv, dp) / np.dot(dp, dp)
                    t = np.clip(t, 0, 1)
                    proj = p1 + t * dp
                    dist = np.linalg.norm(v - proj)
                    if dist < min_dist:
                        min_dist = dist
                pen += min_dist * min_dist
        for v in hex2_verts:
            if point_in_hexagon(v, hex1_verts):
                min_dist = np.inf
                for i in range(6):
                    p1 = hex1_verts[i]
                    p2 = hex1_verts[(i + 1) % 6]
                    dp = p2 - p1
                    dv = v - p1
                    t = np.dot(dv, dp) / np.dot(dp, dp)
                    t = np.clip(t, 0, 1)
                    proj = p1 + t * dp
                    dist = np.linalg.norm(v - proj)
                    if dist < min_dist:
                        min_dist = dist
                pen += min_dist * min_dist
        return pen

    def total_penalty(vars, s):
        # vars: 24 floats (12 x, 12 y)
        # Angles are fixed at 0
        centers = vars.reshape(12, 2)
        
        outer_verts = get_hex_vertices(0, 0, s, 0)
        
        inner_verts_list = []
        for i in range(12):
            cx, cy = centers[i]
            inner_verts_list.append(get_hex_vertices(cx, cy, 1, 0))
            
        pen = 0.0
        # Containment
        for i in range(12):
            pen += containment_penalty(inner_verts_list[i], outer_verts)
            
        # Overlap
        for i in range(12):
            for j in range(i + 1, 12):
                pen += sat_overlap_penalty(inner_verts_list[i], inner_verts_list[j])
                
        return pen

    def objective(vars):
        # s is captured in closure
        return total_penalty(vars, current_s)

    # Binary search for s
    s_low = np.sqrt(12)
    s_high = 6.0
    best_s = s_high
    best_vars = None
    
    # Initial guess for vars
    # Place them in a grid
    init_centers = []
    # Simple grid
    for i in range(4):
        for j in range(3):
            x = (i - 1.5) * 1.8
            y = (j - 1) * 1.8
            init_centers.append([x, y])
    init_vars = np.array(init_centers).flatten()
    
    # Coarse search
    for _ in range(20):
        if time.time() > deadline:
            break
        s_mid = (s_low + s_high) / 2
        current_s = s_mid
        
        # Run optimizer
        # Use multiple restarts
        success = False
        best_pen = np.inf
        best_local_vars = init_vars
        
        for restart in range(3):
            if time.time() > deadline:
                break
            if restart == 0:
                x0 = init_vars
            else:
                x0 = init_vars + np.random.normal(0, 0.5, 24)
                
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
                pen = res.fun
                if pen < best_pen:
                    best_pen = pen
                    best_local_vars = res.x
                if pen < 1e-6:
                    success = True
                    break
            except Exception:
                pass
                
        if success:
            s_high = s_mid
            best_s = s_mid
            best_vars = best_local_vars
        else:
            s_low = s_mid

    # Refine if time permits
    if best_vars is not None and time.time() < deadline:
        # Try to optimize angles too?
        # For now, return the fixed-angle solution
        pass

    # Construct output
    if best_vars is None:
        # Fallback to seed-like solution if optimization failed
        # But we should try to return something valid
        # Use the last tried s_mid?
        # Let's just use a large s to be safe
        inner_hex_data = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0], [-1.25, -2.17, 0],
            [1.25, -2.17, 0], [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0], [0, -4, 0]
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    centers = best_vars.reshape(12, 2)
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0.0
        
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
