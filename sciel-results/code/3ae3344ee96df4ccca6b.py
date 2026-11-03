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

def compute_objective(params, N):
    """
    Compute objective function: outer side length.
    params: array of shape (2*N + 2) = [x1,y1,angle1, ..., xN,yN,angleN, outer_side]
    Actually, let's separate: inner positions and outer side.
    We'll optimize inner positions and outer side together.
    """
    # params layout: for each of N hexagons: x, y, angle_deg
    # Then outer: cx, cy, side, angle_deg
    # Total: 3*N + 4
    
    inner_params = params[:3*N]
    outer_params = params[3*N:]
    
    outer_cx, outer_cy, outer_side, outer_angle = outer_params
    
    if outer_side <= 0:
        return 1e10
    
    # Build outer polygon
    outer_poly = outer_hexagon_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    
    # Check if outer hexagon is valid (non-degenerate)
    # Check each inner hexagon
    inner_polys = []
    for i in range(N):
        ix, iy, iangle = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
        poly = hexagon_vertices(ix, iy, 1.0, iangle)
        inner_polys.append(poly)
        
        # Check if inner hexagon is inside outer
        if not check_inside(poly, outer_poly):
            # Penalize heavily
            return outer_side + 1000.0
    
    # Check pairwise non-intersection
    for i in range(N):
        for j in range(i+1, N):
            if polygons_intersect(inner_polys[i], inner_polys[j]):
                return outer_side + 1000.0
    
    return outer_side

def construct():
    """
    Construct 11 unit hexagons packed in a regular hexagon, minimizing outer side length.
    Returns: (inner, center, side, angle)
    """
    N_local = 11
    
    # Initial guess: asymmetric "defect" cluster
    # Use a 1-2-3-2-1 base with one hexagon rotated 30° to break symmetry
    sqrt3 = math.sqrt(3)
    
    # Base positions for 11 hexagons in a compact arrangement
    # Center + 6 around + 4 more in asymmetric pattern
    init = []
    
    # Center hexagon
    init.append((0.0, 0.0, 0.0))
    
    # 6 surrounding hexagons (hexagonal lattice)
    for i in range(6):
        a = i * math.pi / 3
        init.append((sqrt3 * math.cos(a), sqrt3 * math.sin(a), 0.0))
    
    # 4 additional hexagons in asymmetric "defect" pattern
    # Place them at positions that fill voids but with slight rotation
    # Use angles offset by 30° from lattice positions
    for i in range(4):
        a = (i + 0.5) * math.pi / 3
        # Slightly smaller radius to create "defect"
        r = 2*sqrt3 * 0.95
        # Add 30° rotation to break symmetry
        rot = 30.0 if i % 2 == 0 else 0.0
        init.append((r * math.cos(a), r * math.sin(a), rot))
    
    # Outer hexagon initial guess
    outer_cx = 0.0
    outer_cy = 0.0
    outer_side = 4.5  # Tighter initial guess
    outer_angle = 0.0
    
    # Build initial parameter vector
    x0 = np.zeros(3*N_local + 4)
    for i in range(N_local):
        x0[3*i] = init[i][0]
        x0[3*i+1] = init[i][1]
        x0[3*i+2] = init[i][2]
    x0[3*N_local] = outer_cx
    x0[3*N_local+1] = outer_cy
    x0[3*N_local+2] = outer_side
    x0[3*N_local+3] = outer_angle
    
    # Define objective
    def objective(params):
        return compute_objective(params, N_local)
    
    # Try optimization with multiple strategies
    best_params = x0.copy()
    best_val = objective(x0)
    
    # Strategy 1: Nelder-Mead with perturbed starts
    np.random.seed(42)
    
    for restart in range(8):
        if restart > 0:
            # Create diverse initial guesses
            x0_perturbed = x0.copy()
            
            # Randomly perturb positions
            noise_scale = 0.3 + 0.2 * restart
            noise = np.random.normal(0, noise_scale, size=3*N_local)
            x0_perturbed[:3*N_local] += noise
            
            # Randomly perturb angles
            angle_noise = np.random.uniform(-15, 15, size=N_local)
            for i in range(N_local):
                x0_perturbed[3*i+2] += angle_noise[i]
            
            # Keep outer side reasonable
            x0_perturbed[3*N_local+2] = max(3.5, 4.0 + 0.3 * restart + np.random.normal(0, 0.3))
        else:
            x0_perturbed = x0
        
        try:
            result = minimize(
                objective, 
                x0_perturbed, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True}
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Strategy 2: Try COBYLA for constrained optimization
    for restart in range(3):
        if restart > 0:
            x0_perturbed = x0.copy()
            noise = np.random.normal(0, 0.2, size=3*N_local)
            x0_perturbed[:3*N_local] += noise
            x0_perturbed[3*N_local+2] = max(3.5, 4.0 + np.random.normal(0, 0.2))
        else:
            x0_perturbed = x0
        
        try:
            result = minimize(
                objective, 
                x0_perturbed, 
                method='COBYLA',
                options={'maxiter': 3000, 'rhobeg': 0.1, 'catol': 1e-5}
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_params = result.x.copy()
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
    outer_poly = outer_hexagon_vertices(outer_cx_best, outer_cy_best, outer_side_best, outer_angle_best)
    inner_polys = []
    for i in range(N_local):
        poly = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        inner_polys.append(poly)
        if not check_inside(poly, outer_poly):
            # If invalid, use a safe fallback
            return [[4.0 * k, 0.0, 0.0] for k in range(-(N_local // 2), N_local - N_local // 2)], (0.0, 0.0), 40.0, 0.0
    
    for i in range(N_local):
        for j in range(i+1, N_local):
            if polygons_intersect(inner_polys[i], inner_polys[j]):
                return [[4.0 * k, 0.0, 0.0] for k in range(-(N_local // 2), N_local - N_local // 2)], (0.0, 0.0), 40.0, 0.0
    
    return inner, (outer_cx_best, outer_cy_best), outer_side_best, outer_angle_best

# EVOLVE-BLOCK-START
# (construct function defined above)
# EVOLVE-BLOCK-END
