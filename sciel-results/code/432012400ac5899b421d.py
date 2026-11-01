"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    def hex_vertices(cx, cy, side, angle_deg):
        theta = np.radians(angle_deg)
        angles = theta + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])
    
    def hex_hull_distance(h1, h2):
        """Approximate minimum distance between two hexagons. Returns negative if overlapping."""
        # Use vertex-to-edge and edge-to-edge checks (simplified: just check if any vertex of one is inside the other)
        # For a quick check, use the minimum distance between any vertex of h1 and any edge of h2, and vice versa
        # Simplified: check if centroids are too close
        c1 = np.mean(h1, axis=0)
        c2 = np.mean(h2, axis=0)
        dist = np.linalg.norm(c1 - c2)
        # Minimum distance between non-overlapping unit hexagons is about 2 (side to side)
        # This is a rough heuristic
        return dist - 2.0  # Negative means likely overlapping
    
    def polygons_disjoint(v1, v2, tol=1e-7):
        """Check if two convex polygons are disjoint using SAT."""
        def project_polygon(vertices, axis):
            dots = np.dot(vertices, axis)
            return np.min(dots), np.max(dots)
        
        def get_axes(vertices):
            axes = []
            for i in range(len(vertices)):
                edge = vertices[(i+1) % len(vertices)] - vertices[i]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    axes.append(normal / norm)
            return axes
        
        axes = get_axes(v1) + get_axes(v2)
        for axis in axes:
            min1, max1 = project_polygon(v1, axis)
            min2, max2 = project_polygon(v2, axis)
            if max1 < min2 - tol or max2 < min1 - tol:
                return True  # Disjoint
        return False  # Overlapping
    
    def overlap_penalty(inner_hexes, penalty_weight=100.0):
        """Penalty for overlapping inner hexagons."""
        penalty = 0.0
        verts_list = [hex_vertices(x, y, 1.0, a) for (x, y, a) in inner_hexes]
        for i in range(len(verts_list)):
            for j in range(i+1, len(verts_list)):
                if not polygons_disjoint(verts_list[i], verts_list[j]):
                    # Compute overlap amount (approximate)
                    c1 = np.mean(verts_list[i], axis=0)
                    c2 = np.mean(verts_list[j], axis=0)
                    dist = np.linalg.norm(c1 - c2)
                    penalty += penalty_weight * max(0, 2.0 - dist) ** 2
        return penalty
    
    def max_required_side(inner_hexes, outer_cx, outer_cy, outer_angle):
        max_s = 0.0
        for (ix, iy, iang) in inner_hexes:
            verts = hex_vertices(ix, iy, 1.0, iang)
            for (vx, vy) in verts:
                dx = vx - outer_cx
                dy = vy - outer_cy
                theta = np.radians(outer_angle)
                cos_t = np.cos(-theta)
                sin_t = np.sin(-theta)
                lx = dx * cos_t - dy * sin_t
                ly = dx * sin_t + dy * cos_t
                for k in range(6):
                    normal_angle = np.radians(30 + 60 * k)
                    nx = np.cos(normal_angle)
                    ny = np.sin(normal_angle)
                    proj = lx * nx + ly * ny
                    if proj > 0:
                        s_req = proj / (np.sqrt(3) / 2)
                        if s_req > max_s:
                            max_s = s_req
        return max_s
    
    def objective(params, outer_angle=0.0):
        inner_hexes = []
        for i in range(N):
            x = params[2*i]
            y = params[2*i+1]
            inner_hexes.append((x, y, 0.0))
        side = max_required_side(inner_hexes, 0.0, 0.0, outer_angle)
        pen = overlap_penalty(inner_hexes)
        return side + pen
    
    # Initial honeycomb arrangement
    sqrt3 = np.sqrt(3)
    initial_params = []
    for row in range(3):
        y = row * sqrt3
        offset = 0.0 if row % 2 == 0 else 0.75
        for col in range(4):
            x = col * 1.5 - 2.25 + offset
            initial_params.append(x)
            initial_params.append(y)
    
    initial_params = np.array(initial_params)
    cx_arr = initial_params[0::2]
    cy_arr = initial_params[1::2]
    initial_params[0::2] = cx_arr - np.mean(cx_arr)
    initial_params[1::2] = cy_arr - np.mean(cy_arr)
    
    best_params = initial_params.copy()
    best_obj = objective(initial_params)
    
    # Nelder-Mead optimization
    try:
        result = minimize(objective, initial_params, method='Nelder-Mead',
                         options={'maxiter': 8000, 'xatol': 1e-5, 'fatol': 1e-7,
                                  'adaptive': True})
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    except Exception:
        pass
    
    if time.time() > deadline - 5.0:
        inner = [[best_params[2*i], best_params[2*i+1], 0.0] for i in range(N)]
        # Verify no overlaps
        verts_list = [hex_vertices(x, y, 1.0, 0.0) for (x, y, _) in inner]
        has_overlap = False
        for i in range(len(verts_list)):
            for j in range(i+1, len(verts_list)):
                if not polygons_disjoint(verts_list[i], verts_list[j]):
                    has_overlap = True
                    break
            if has_overlap:
                break
        if has_overlap:
            # Fall back to seed
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0
        return inner, (0.0, 0.0), best_obj, 0.0
    
    # Try outer angle 30
    try:
        result = minimize(lambda p: objective(p, 30.0), initial_params, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-7,
                                  'adaptive': True})
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    except Exception:
        pass
    
    inner = [[best_params[2*i], best_params[2*i+1], 0.0] for i in range(N)]
    verts_list = [hex_vertices(x, y, 1.0, 0.0) for (x, y, _) in inner]
    has_overlap = False
    for i in range(len(verts_list)):
        for j in range(i+1, len(verts_list)):
            if not polygons_disjoint(verts_list[i], verts_list[j]):
                has_overlap = True
                break
        if has_overlap:
            break
    if has_overlap:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    return inner, (0.0, 0.0), best_obj, 0.0

# EVOLVE-BLOCK-END
