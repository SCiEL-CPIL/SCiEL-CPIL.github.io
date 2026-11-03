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

def check_inside(inner_hex, outer_poly):
    """Check if all vertices of inner hexagon are inside outer polygon."""
    for v in inner_hex:
        if not point_in_polygon(v[0], v[1], outer_poly):
            return False
    return True

def compute_objective(params, N):
    """
    Compute objective function: outer side length + penalties.
    params: [x1, y1, a1, ..., xN, yN, aN, outer_side]
    Outer hexagon is fixed at (0,0) with angle 0.
    """
    inner_params = params[:3*N]
    outer_side = params[3*N]
    
    if outer_side <= 0:
        return 1e10
        
    # Build outer polygon (fixed center and angle)
    outer_poly = hexagon_vertices(0.0, 0.0, outer_side, 0.0)
    
    # Check each inner hexagon
    inner_polys = []
    penalty = 0.0
    for i in range(N):
        ix, iy, iangle = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
        poly = hexagon_vertices(ix, iy, 1.0, iangle)
        inner_polys.append(poly)
        
        # Check if inner hexagon is inside outer
        if not check_inside(poly, outer_poly):
            # Calculate violation distance roughly for smooth penalty
            # If a vertex is outside, add penalty
            for v in poly:
                if not point_in_polygon(v[0], v[1], outer_poly):
                    penalty += 10.0
                    # Estimate distance to boundary for gradient
                    # Simple heuristic: if outside, push it in? 
                    # For now, just a large penalty
            # To make it differentiable, we can use the distance to the nearest edge if outside
            # But for Nelder-Mead, a hard penalty is often sufficient if we start close.
            # Let's use a softer penalty: sum of squared distances of vertices outside
            for v in poly:
                if not point_in_polygon(v[0], v[1], outer_poly):
                    # Find distance to closest point on outer boundary
                    # This is expensive. Let's just use a constant penalty per outside vertex.
                    pass 
            penalty += 50.0 # Base penalty if any part is outside
            
    # Check pairwise non-intersection
    for i in range(N):
        for j in range(i+1, N):
            if polygons_intersect(inner_polys[i], inner_polys[j]):
                penalty += 50.0
    
    return outer_side + penalty

def construct():
    """
    Construct 11 unit hexagons packed in a regular hexagon, minimizing outer side length.
    Returns: (inner, center, side, angle)
    """
    N_local = 11
    sqrt3 = math.sqrt(3)
    
    # Asymmetric "defect" cluster initial guess
    # Center at (0,0)
    # 6 neighbors at distance sqrt(3)
    # 4 more placed to break symmetry
    
    init = []
    # Center
    init.append((0.0, 0.0, 0.0))
    # 6 around
    for i in range(6):
        a = i * math.pi / 3
        init.append((sqrt3 * math.cos(a), sqrt3 * math.sin(a), 0.0))
    # 4 more: placed in the "gaps" of the second ring, but asymmetrically
    # Angles: 30, 150, 210, 330 degrees?
    # Distance: 2*sqrt(3) * cos(30) = 3? No, distance to next ring in hex lattice is 3.
    # Let's try distance 2.5 and angles 30, 150, 270, 30? 
    # Let's just use a known good starting point for 11 hexagons.
    # A common packing is a 3-4-3 or similar strip, but hexagonal is tighter.
    # Let's try:
    # Row 1: 1
    # Row 2: 2
    # Row 3: 3
    # Row 4: 2
    # Row 5: 1
    # This is 9.
    # Add 2 more to the sides?
    
    # Let's use a random restart strategy with a good seed.
    np.random.seed(42)
    
    # Create initial parameters
    x0 = np.zeros(3*N_local + 1)
    for i in range(N_local):
        x0[3*i] = init[i][0]
        x0[3*i+1] = init[i][1]
        x0[3*i+2] = init[i][2]
    x0[3*N_local] = 5.0 # Outer side initial guess
    
    # Define objective
    def objective(params):
        return compute_objective(params, N_local)
    
    best_params = x0.copy()
    best_val = objective(x0)
    
    # Optimization Strategy:
    # 1. Try L-BFGS-B with bounds? No bounds needed if penalty is high.
    # 2. Try Nelder-Mead for robustness.
    
    for restart in range(10):
        if restart == 0:
            x_curr = x0.copy()
        else:
            # Perturb
            x_curr = x0.copy()
            noise = np.random.normal(0, 0.3, size=3*N_local)
            x_curr[:3*N_local] += noise
            x_curr[3*N_local] = max(4.0, x0[3*N_local] + np.random.normal(0, 0.5))
            
        try:
            # L-BFGS-B
            res_lbfgs = minimize(
                objective, 
                x_curr, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            if res_lbfgs.fun < best_val:
                best_val = res_lbfgs.fun
                best_params = res_lbfgs.x
            
            # Nelder-Mead on the best so far (or LBFGS result if it was better)
            x_nm = res_lbfgs.x if res_lbfgs.fun < objective(x_curr) else x_curr
            res_nm = minimize(
                objective, 
                x_nm, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_params = res_nm.x
                
        except Exception as e:
            pass
    
    # Extract best parameters
    inner = []
    for i in range(N_local):
        ix = best_params[3*i]
        iy = best_params[3*i+1]
        iangle = best_params[3*i+2]
        inner.append([ix, iy, iangle])
    
    outer_side_best = best_params[3*N_local]
    
    # Final validation
    outer_poly = hexagon_vertices(0.0, 0.0, outer_side_best, 0.0)
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
        # Fallback to a known valid configuration (even if not optimal)
        # Simple grid: 11 hexagons in a row? No, too wide.
        # 3x4 grid? 
        # Let's just return a very large hexagon with hexagons spaced far apart.
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N_local // 2), N_local - N_local // 2)], (0.0, 0.0), 40.0, 0.0

    return inner, (0.0, 0.0), outer_side_best, 0.0

# EVOLVE-BLOCK-START
# (construct function defined above)
# EVOLVE-BLOCK-END
