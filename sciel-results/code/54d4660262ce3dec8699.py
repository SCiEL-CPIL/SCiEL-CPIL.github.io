"""
Improved packing of 11 unit hexagons in a regular hexagon.
Uses a multi-start Nelder-Mead optimization with a penalty-based objective function.
Initial configurations are based on hexagonal lattice clusters with defects.
"""
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
    Compute objective function: outer side length + penalties for violations.
    """
    inner_params = params[:3*N]
    outer_params = params[3*N:]
    
    outer_cx, outer_cy, outer_side, outer_angle = outer_params
    
    if outer_side <= 0:
        return 1e10
    
    outer_poly = hexagon_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    
    penalty = 0.0
    inner_polys = []
    
    for i in range(N):
        ix, iy, iangle = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
        poly = hexagon_vertices(ix, iy, 1.0, iangle)
        inner_polys.append(poly)
        
        if not check_inside(poly, outer_poly):
            penalty += 50.0
    
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
    
    # Define several initial configurations
    # 1. Hexagonal cluster (7) + 4 corners
    init1 = [
        (0.0, 0.0, 0.0),
        (sqrt3, 0.0, 0.0),
        (sqrt3/2, sqrt3/2, 0.0),
        (-sqrt3/2, sqrt3/2, 0.0),
        (-sqrt3, 0.0, 0.0),
        (-sqrt3/2, -sqrt3/2, 0.0),
        (sqrt3/2, -sqrt3/2, 0.0),
        (2*sqrt3, 0.0, 0.0),
        (sqrt3, sqrt3, 0.0),
        (0.0, 2*sqrt3, 0.0),
        (-sqrt3, sqrt3, 0.0),
    ]
    
    # 2. Linear arrangement (fallback)
    init2 = [(4.0 * k, 0.0, 0.0) for k in range(-5, 6)]
    
    # 3. Asymmetric defect cluster
    # Rotate one of the outer hexagons by 30 degrees
    init3 = [p for p in init1]
    init3[7] = (2*sqrt3, 0.0, 30.0) # Rotate the rightmost one
    
    initial_guesses = [init1, init2, init3]
    
    best_params = None
    best_val = float('inf')
    
    np.random.seed(42)
    
    for init_pos in initial_guesses:
        x0 = np.zeros(3*N_local + 4)
        for i in range(N_local):
            x0[3*i] = init_pos[i][0]
            x0[3*i+1] = init_pos[i][1]
            x0[3*i+2] = init_pos[i][2]
        
        # Estimate outer side
        # Max distance from origin
        max_r = max(math.sqrt(p[0]**2 + p[1]**2) for p in init_pos)
        outer_side_init = max_r + 2.0
        x0[3*N_local] = 0.0
        x0[3*N_local+1] = 0.0
        x0[3*N_local+2] = outer_side_init
        x0[3*N_local+3] = 0.0
        
        # Multi-start for this initial guess
        for restart in range(5):
            if restart == 0:
                x_current = x0
            else:
                x_current = x0.copy()
                noise = np.random.normal(0, 0.3, size=3*N_local)
                x_current[:3*N_local] += noise
                x_current[3*N_local+2] = outer_side_init + np.random.normal(0, 0.2)
            
            try:
                result = minimize(
                    lambda p: compute_objective(p, N_local),
                    x_current,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5}
                )
                val = result.fun
                if val < best_val:
                    best_val = val
                    best_params = result.x
            except Exception:
                pass
    
    # Extract best parameters
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
    
    # Ensure outer side is positive
    outer_side_best = max(outer_side_best, 3.0)
    
    # Final validation
    outer_poly = hexagon_vertices(outer_cx_best, outer_cy_best, outer_side_best, outer_angle_best)
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
        # Fallback to a known valid packing
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N_local // 2), N_local - N_local // 2)]
        outer_cx_best = 0.0
        outer_cy_best = 0.0
        outer_side_best = 40.0
        outer_angle_best = 0.0
    
    return inner, (outer_cx_best, outer_cy_best), outer_side_best, outer_angle_best

# EVOLVE-BLOCK-START
# (construct function defined above)
# EVOLVE-BLOCK-END
