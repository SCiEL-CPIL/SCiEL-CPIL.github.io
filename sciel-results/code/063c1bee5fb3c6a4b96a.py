"""Rewritten program for packing 11 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import math

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        a = angle_rad + i * math.pi / 3
        x = cx + side * math.cos(a)
        y = cy + side * math.sin(a)
        vertices.append((x, y))
    return vertices

def point_in_polygon(px, py, polygon):
    """Ray casting algorithm to check if point is inside polygon."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def segment_intersection(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def polygons_intersect(poly1, poly2):
    """Check if two polygons intersect (including one containing the other)."""
    # Check if any vertex of poly1 is inside poly2
    for v in poly1:
        if point_in_polygon(v[0], v[1], poly2):
            return True
    # Check if any vertex of poly2 is inside poly1
    for v in poly2:
        if point_in_polygon(v[0], v[1], poly1):
            return True
    # Check for edge intersections
    n1 = len(poly1)
    n2 = len(poly2)
    for i in range(n1):
        for j in range(n2):
            if segment_intersection(poly1[i], poly1[(i+1)%n1], poly2[j], poly2[(j+1)%n2]):
                return True
    return False

def outer_hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of outer regular hexagon."""
    return hexagon_vertices(cx, cy, side, angle_deg)

def check_inside(inner_hex, outer_poly):
    """Check if all vertices of inner hexagon are inside outer polygon."""
    for v in inner_hex:
        if not point_in_polygon(v[0], v[1], outer_poly):
            return False
    return True

def signed_distance_point_to_segment(px, py, x1, y1, x2, y2):
    """
    Calculate signed distance from point (px, py) to segment (x1,y1)-(x2,y2).
    Positive if point is on the left of the directed segment.
    """
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx*dx + dy*dy
    
    if length_sq == 0:
        return 0.0
    
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    
    if t < 0:
        dist = math.hypot(px - x1, py - y1)
        cross = dx * (py - y1) - dy * (px - x1)
        return -dist if cross < 0 else dist
    elif t > 1:
        dist = math.hypot(px - x2, py - y2)
        cross = dx * (py - y2) - dy * (px - x2)
        return -dist if cross < 0 else dist
    else:
        cross = dx * (py - y1) - dy * (px - x1)
        dist = abs(cross) / math.sqrt(length_sq)
        return -dist if cross < 0 else dist

def signed_distance_point_to_polygon(px, py, polygon):
    """
    Calculate signed distance from point to polygon.
    Positive if inside, negative if outside.
    """
    if point_in_polygon(px, py, polygon):
        min_dist = float('inf')
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i+1)%n]
            d = signed_distance_point_to_segment(px, py, x1, y1, x2, y2)
            if d < min_dist:
                min_dist = d
        return min_dist
    else:
        min_dist = float('inf')
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i+1)%n]
            d = signed_distance_point_to_segment(px, py, x1, y1, x2, y2)
            abs_d = abs(d)
            if abs_d < min_dist:
                min_dist = abs_d
        return -min_dist

def smooth_overlap_penalty(poly1, poly2):
    """
    Calculate a smooth penalty for overlap between two polygons.
    """
    penalty = 0.0
    for v in poly1:
        d = signed_distance_point_to_polygon(v[0], v[1], poly2)
        if d > 0:
            penalty += d * d
    for v in poly2:
        d = signed_distance_point_to_polygon(v[0], v[1], poly1)
        if d > 0:
            penalty += d * d
    return penalty

def smooth_containment_penalty(inner_poly, outer_poly):
    """
    Calculate smooth penalty if inner_poly is not strictly inside outer_poly.
    """
    penalty = 0.0
    for v in inner_poly:
        d = signed_distance_point_to_polygon(v[0], v[1], outer_poly)
        if d < 0:
            penalty += (-d) * (-d)
    return penalty

def compute_objective(params, N):
    """
    Compute objective function: outer side length + penalties.
    """
    inner_params = params[:3*N]
    outer_params = params[3*N:]
    
    outer_cx, outer_cy, outer_side, outer_angle = outer_params
    
    if outer_side <= 0:
        return 1e10
    
    outer_poly = outer_hexagon_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    
    inner_polys = []
    total_penalty = 0.0
    
    for i in range(N):
        ix, iy, iangle = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
        poly = hexagon_vertices(ix, iy, 1.0, iangle)
        inner_polys.append(poly)
        total_penalty += smooth_containment_penalty(poly, outer_poly)
        
    for i in range(N):
        for j in range(i+1, N):
            total_penalty += smooth_overlap_penalty(inner_polys[i], inner_polys[j])
            
    return outer_side + 10.0 * total_penalty

def construct():
    """
    Construct 11 unit hexagons packed in a regular hexagon, minimizing outer side length.
    Returns: (inner, center, side, angle)
    """
    N_local = 11
    
    sqrt3 = math.sqrt(3)
    
    # Topological initialization:
    # 1. Center hexagon at (0,0)
    # 2. 6 hexagons in a ring around center (distance sqrt(3))
    # 3. 4 hexagons in an asymmetric outer layer to break symmetry ("defect" cluster)
    
    init_positions = []
    
    # Center
    init_positions.append((0.0, 0.0, 0.0))
    
    # Ring 1: 6 hexagons
    for i in range(6):
        a = i * math.pi / 3
        x = sqrt3 * math.cos(a)
        y = sqrt3 * math.sin(a)
        init_positions.append((x, y, 0.0))
        
    # Ring 2: 4 hexagons placed asymmetrically
    # In a perfect 1-2-3-2-1 lattice, we would have more, but we only have 4 left.
    # We place them at the "corners" of the hexagonal lattice but rotate some to fill gaps.
    # Distance for second ring in hex lattice is 2*sqrt(3)
    r2 = 2.0 * sqrt3
    
    # Angles for the 4 outer hexagons, slightly perturbed from symmetric positions
    # Symmetric positions would be at 0, 90, 180, 270 (or similar) but hex lattice is 60deg symmetric.
    # Let's place them at angles: 0, 60, 120, 180 but with varying rotations.
    # Actually, let's just pick 4 spots that are far apart and let the optimizer adjust.
    # Spots:
    # 1. Top right (30 deg)
    # 2. Top left (150 deg)
    # 3. Bottom left (210 deg)
    # 4. Bottom right (330 deg)
    
    angles = [30, 150, 210, 330]
    rotations = [0.0, 0.0, 0.0, 30.0] # One rotated to break symmetry
    
    for i in range(4):
        a = math.radians(angles[i])
        x = r2 * math.cos(a)
        y = r2 * math.sin(a)
        init_positions.append((x, y, rotations[i]))
        
    outer_cx = 0.0
    outer_cy = 0.0
    outer_side = 4.5 # Initial guess
    outer_angle = 0.0
    
    x0 = np.zeros(3*N_local + 4)
    for i in range(N_local):
        x0[3*i] = init_positions[i][0]
        x0[3*i+1] = init_positions[i][1]
        x0[3*i+2] = init_positions[i][2]
    x0[3*N_local] = outer_cx
    x0[3*N_local+1] = outer_cy
    x0[3*N_local+2] = outer_side
    x0[3*N_local+3] = outer_angle
    
    def objective(params):
        return compute_objective(params, N_local)
        
    best_params = x0.copy()
    best_val = objective(x0)
    
    # Use SLSQP or L-BFGS-B for better gradient handling
    # L-BFGS-B is good for large N, SLSQP can handle constraints but we use penalty.
    # Let's try L-BFGS-B first with multiple restarts.
    
    methods = ['L-BFGS-B', 'Nelder-Mead']
    options_lbfgsb = {'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-5}
    options_nm = {'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5}
    
    # Restart 1: Asymmetric init
    try:
        result = minimize(objective, x0, method='L-BFGS-B', options=options_lbfgsb)
        if result.fun < best_val:
            best_val = result.fun
            best_params = result.x
    except Exception:
        pass
        
    # Restart 2: Perturbed init
    np.random.seed(42)
    x0_perturbed = x0.copy()
    for i in range(N_local):
        x0_perturbed[3*i] += np.random.uniform(-0.2, 0.2)
        x0_perturbed[3*i+1] += np.random.uniform(-0.2, 0.2)
        x0_perturbed[3*i+2] += np.random.uniform(-10, 10)
    x0_perturbed[3*N_local+2] = 4.2
        
    try:
        result2 = minimize(objective, x0_perturbed, method='L-BFGS-B', options=options_lbfgsb)
        if result2.fun < best_val:
            best_val = result2.fun
            best_params = result2.x
    except Exception:
        pass
        
    # Restart 3: Nelder-Mead on best found so far (polishing)
    try:
        result3 = minimize(objective, best_params, method='Nelder-Mead', options=options_nm)
        if result3.fun < best_val:
            best_val = result3.fun
            best_params = result3.x
    except Exception:
        pass
        
    inner = []
    for i in range(N_local):
        ix = best_params[3*i]
        iy = best_params[3*i+1]
        iangle = best_params[3*i+2]
        inner.append([ix, iy, iangle])
    
    outer_cx_best = best_params[3*N_local]
    outer_cy_best = best_params[3*N_local+1]
    outer_side_best = best_params[3*N_local+2]
    outer_angle_best = best_params[3*N_local+3]
    
    outer_side_best = max(outer_side_best, 3.0)
    
    outer_poly = outer_hexagon_vertices(outer_cx_best, outer_cy_best, outer_side_best, outer_angle_best)
    inner_polys = []
    valid = True
    for i in range(N_local):
        poly = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        inner_polys.append(poly)
        if not check_inside(poly, outer_poly):
            valid = False
            break
    
    if valid:
        for i in range(N_local):
            for j in range(i+1, N_local):
                if polygons_intersect(inner_polys[i], inner_polys[j]):
                    valid = False
                    break
            if not valid:
                break
                
    if not valid:
        fallback_inner = []
        for i in range(N_local):
            row = i // 3
            col = i % 3
            x = col * 2.0 - 2.0
            y = row * 1.8 - 1.8
            fallback_inner.append([x, y, 0.0])
        return fallback_inner, (0.0, 0.0), 10.0, 0.0
        
    return inner, (outer_cx_best, outer_cy_best), outer_side_best, outer_angle_best

# EVOLVE-BLOCK-START
# (construct function defined above)
# EVOLVE-BLOCK-END
