import numpy as np
import time
from scipy.optimize import minimize

N = 11

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append([cx + s * np.cos(theta), cy + s * np.sin(theta)])
    return np.array(verts)

def point_in_poly(px, py, poly):
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def seg_intersect(p1, p2, p3, p4):
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

def hexagons_disjoint(v1, v2):
    """Check if two hexagons (vertex arrays) are disjoint (no interior intersection)."""
    for v in v1:
        if point_in_poly(v[0], v[1], v2):
            return False
    for v in v2:
        if point_in_poly(v[0], v[1], v1):
            return False
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j+1)%6]
            if seg_intersect(p1, p2, p3, p4):
                return False
    return True

def min_enclosing_hex_side(points, outer_angle_deg):
    """
    Compute the minimum side length of a regular hexagon with given rotation
    that contains all points. The center is optimized.
    """
    a = np.radians(outer_angle_deg)
    # Normals for the 6 sides
    normals = []
    for i in range(6):
        theta = a + np.pi/6 + i * np.pi / 3
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals)
    
    # To find the minimal side length, we need to find the center (cx, cy) that minimizes
    # max_i (max_p dot(p-c, n_i) - min_p dot(p-c, n_i)) / sqrt(3)
    # Let w_i(c) = max_p (p-c).n_i - min_p (p-c).n_i = max_p p.n_i - c.n_i - (min_p p.n_i - c.n_i) = max_p p.n_i - min_p p.n_i
    # Wait, the width in direction n_i is independent of the center c!
    # The width of the set of points in direction n is fixed.
    # The side length s required is max_i (width_i) / sqrt(3).
    # However, the hexagon must *contain* the points.
    # The condition for point p to be in hexagon centered at c with side s is:
    # dot(p-c, n_i) <= s * sqrt(3)/2 for all i.
    # This implies:
    # dot(p, n_i) - dot(c, n_i) <= s * sqrt(3)/2
    # dot(p, n_i) - s * sqrt(3)/2 <= dot(c, n_i)
    # So for each i, dot(c, n_i) must be in [max_p dot(p, n_i) - s*sqrt(3)/2, min_p dot(p, n_i) + s*sqrt(3)/2]
    # Let P_max_i = max_p dot(p, n_i) and P_min_i = min_p dot(p, n_i).
    # We need to find c and s such that for all i:
    # P_max_i - s*sqrt(3)/2 <= dot(c, n_i) <= P_min_i + s*sqrt(3)/2
    # This is a feasibility problem. We want to minimize s.
    # This is equivalent to finding the smallest regular hexagon (fixed orientation) containing the points.
    # The optimal center c is the center of the intersection of the strips.
    # A simpler numerical approach:
    # The side length is determined by the maximum projection width.
    # Width_i = P_max_i - P_min_i.
    # s_min = max_i(Width_i) / sqrt(3).
    # But we must ensure there exists a center c.
    # The constraints on c are:
    # dot(c, n_i) >= P_max_i - s*sqrt(3)/2
    # dot(c, n_i) <= P_min_i + s*sqrt(3)/2
    # Let L_i = P_max_i - s*sqrt(3)/2 and U_i = P_min_i + s*sqrt(3)/2.
    # Note that L_i <= U_i is equivalent to P_max_i - P_min_i <= s*sqrt(3), i.e., Width_i <= s*sqrt(3).
    # So if s >= max(Width_i)/sqrt(3), the intervals [L_i, U_i] are non-empty.
    # We need to check if the intersection of the half-planes defined by these constraints is non-empty.
    # Since the normals n_i form a regular hexagon set, the intersection of these strips is a hexagon (or unbounded).
    # If the intersection is non-empty, the minimal s is max(Width_i)/sqrt(3).
    # Is it always possible to find such a c?
    # Yes, for a convex set, the minimal enclosing regular polygon with fixed orientation exists.
    # The side length is indeed max_i (Width_i) / sqrt(3).
    # The center c can be any point in the intersection.
    # For the purpose of the objective function, we just need the side length.
    
    proj_max = np.max(points @ normals.T, axis=0)
    proj_min = np.min(points @ normals.T, axis=0)
    widths = proj_max - proj_min
    s = np.max(widths) / np.sqrt(3)
    return s

def objective(params):
    # params: 11 * 3 (x, y, angle) + 3 (cx, cy, angle)
    # Total 36 parameters
    inner_params = params[:N*3].reshape(N, 3)
    outer_params = params[N*3:]
    outer_cx, outer_cy, outer_angle = outer_params
    
    all_points = []
    inner_verts = []
    for k in range(N):
        x, y, ang = inner_params[k]
        verts = hex_vertices(x, y, 1.0, ang)
        inner_verts.append(verts)
        all_points.extend(verts)
    
    all_points = np.array(all_points)
    
    # Calculate side length
    s = min_enclosing_hex_side(all_points, outer_angle)
    
    # Check overlaps
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                penalty += 100.0
    
    return s + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = np.sqrt(3)
    
    # Initial lattice positions
    hex_positions_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2),
    ]
    
    initial_centers = []
    for q, r in hex_positions_axial:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        initial_centers.append((x, y))
    
    initial_params = []
    for x, y in initial_centers:
        initial_params.extend([x, y, 0.0])
    
    # Initial outer hexagon params
    # Center of mass of initial points
    all_verts_init = []
    for x, y in initial_centers:
        all_verts_init.extend(hex_vertices(x, y, 1.0, 0.0))
    all_verts_init = np.array(all_verts_init)
    cx0 = all_verts_init[:, 0].mean()
    cy0 = all_verts_init[:, 1].mean()
    angle0 = 0.0
    
    initial_params.extend([cx0, cy0, angle0])
    initial_params = np.array(initial_params)
    
    best_result = None
    best_score = float('inf')
    
    # Run optimization
    try:
        result = minimize(objective, initial_params, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
        
        if result.fun < best_score:
            best_score = result.fun
            best_result = result.x
            
        # Check time
        if time.time() > deadline - 20:
            pass
        else:
            # Try a few random restarts if time permits
            for _ in range(3):
                if time.time() > deadline - 20:
                    break
                
                # Perturb initial parameters
                perturbed = initial_params + np.random.normal(0, 0.5, size=initial_params.shape)
                # Keep angles in a reasonable range
                for i in range(N):
                    perturbed[i*3+2] = np.random.uniform(-30, 30)
                perturbed[N*3+2] = np.random.uniform(-30, 30)
                
                res = minimize(objective, perturbed, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
                
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res.x
                    
    except Exception as e:
        # Fallback to initial lattice if optimization fails
        best_result = initial_params
        best_score = objective(initial_params)

    if best_result is None:
        best_result = initial_params
        best_score = objective(best_result)

    # Extract final parameters
    inner_params = best_result[:N*3].reshape(N, 3)
    outer_params = best_result[N*3:]
    outer_cx, outer_cy, outer_angle = outer_params
    
    inner = []
    for k in range(N):
        x, y, ang = inner_params[k]
        inner.append([float(x), float(y), float(ang)])
    
    # Final verification and adjustment
    # Recalculate the exact side length needed
    all_verts = []
    for x, y, ang in inner_params:
        all_verts.extend(hex_vertices(x, y, 1.0, ang))
    all_verts = np.array(all_verts)
    
    final_side = min_enclosing_hex_side(all_verts, outer_angle)
    
    # Add a tiny epsilon for safety
    final_side += 1e-9
    
    return inner, (float(outer_cx), float(outer_cy)), float(final_side), float(outer_angle)
