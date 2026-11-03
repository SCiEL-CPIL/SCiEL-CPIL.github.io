"""Topological graph optimization for packing 11 unit hexagons in a regular hexagon."""
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
    for v in poly1:
        if point_in_polygon(v[0], v[1], poly2):
            return True
    for v in poly2:
        if point_in_polygon(v[0], v[1], poly1):
            return True
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

def compute_min_outer_side(inner_positions, outer_angle=0.0):
    """
    Compute the minimum outer hexagon side length needed to contain all inner hexagons.
    inner_positions: list of (x, y, angle_deg) for N hexagons
    Returns: (min_side, optimal_cx, optimal_cy)
    """
    N_local = len(inner_positions)
    
    # Collect all vertices of all inner hexagons
    all_vertices = []
    for ix, iy, iangle in inner_positions:
        verts = hexagon_vertices(ix, iy, 1.0, iangle)
        all_vertices.extend(verts)
    
    all_vertices = np.array(all_vertices)
    
    # The outer hexagon has 6 sides. For a regular hexagon with center (cx, cy), side s, angle theta:
    # The 6 outward normals are at angles theta, theta+60, theta+120, theta+180, theta+240, theta+300
    # A point (px, py) is inside if for all 6 directions, the projection is within the support
    
    def required_side_for_direction(vertices, cx, cy, theta):
        """Compute required side length for a given outward normal direction."""
        # Normal direction
        nx = math.cos(theta)
        ny = math.sin(theta)
        
        # The constraint is: for all vertices, (px - cx)*nx + (py - cy)*ny <= s * sqrt(3)/2
        # where s * sqrt(3)/2 is the distance from center to side (apothem)
        
        # So s >= 2/sqrt(3) * max_i [(px_i - cx)*nx + (py_i - cy)*ny]
        
        projections = (vertices[:, 0] - cx) * nx + (vertices[:, 1] - cy) * ny
        max_proj = np.max(projections)
        
        if max_proj < 0:
            return 0.0
        
        return 2.0 / math.sqrt(3) * max_proj
    
    # We need to find cx, cy that minimize the maximum required side over all 6 directions
    def outer_side_for_center(cx, cy, outer_angle_deg):
        outer_angle_rad = math.radians(outer_angle_deg)
        max_side = 0.0
        for k in range(6):
            theta = outer_angle_rad + k * math.pi / 3
            side_k = required_side_for_direction(all_vertices, cx, cy, theta)
            max_side = max(max_side, side_k)
        return max_side
    
    # Optimize over cx, cy to minimize the outer side
    def objective_center(params):
        cx, cy = params
        return outer_side_for_center(cx, cy, outer_angle)
    
    # Initial guess: centroid of all vertices
    cx0 = np.mean(all_vertices[:, 0])
    cy0 = np.mean(all_vertices[:, 1])
    
    try:
        result = minimize(objective_center, [cx0, cy0], method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        best_cx, best_cy = result.x
        best_side = outer_side_for_center(best_cx, best_cy, outer_angle)
    except Exception:
        best_cx, best_cy = cx0, cy0
        best_side = outer_side_for_center(cx0, cy0, outer_angle)
    
    return best_side, best_cx, best_cy

def compute_objective(params, N):
    """
    Compute objective function: outer side length.
    params: array of shape (3*N,) = [x1,y1,angle1, ..., xN,yN,angleN]
    """
    inner_positions = []
    for i in range(N):
        ix, iy, iangle = params[3*i], params[3*i+1], params[3*i+2]
        inner_positions.append((ix, iy, iangle))
    
    # Check pairwise non-intersection first
    inner_polys = []
    for ix, iy, iangle in inner_positions:
        poly = hexagon_vertices(ix, iy, 1.0, iangle)
        inner_polys.append(poly)
    
    for i in range(N):
        for j in range(i+1, N):
            if polygons_intersect(inner_polys[i], inner_polys[j]):
                # Return a large penalty
                return 1000.0
    
    # Compute minimum outer side
    min_side, _, _ = compute_min_outer_side(inner_positions)
    return min_side

def construct():
    """
    Construct 11 unit hexagons packed in a regular hexagon, minimizing outer side length.
    Returns: (inner, center, side, angle)
    """
    N_local = 11
    sqrt3 = math.sqrt(3)
    
    # Design an asymmetric "defect" cluster:
    # Start with a 1-2-3-2-1 hexagonal lattice (9 hexagons)
    # Add 2 more hexagons, one rotated 30° to fill a void
    # This breaks the symmetry and may allow tighter packing
    
    # Base 1-2-3-2-1 arrangement
    base_positions = [
        (0.0, 2*sqrt3, 0.0),           # Row 0: 1 hex
        (-sqrt3/2, sqrt3, 0.0),         # Row 1: 2 hexes
        (sqrt3/2, sqrt3, 0.0),
        (-sqrt3, 0.0, 0.0),             # Row 2: 3 hexes
        (0.0, 0.0, 0.0),
        (sqrt3, 0.0, 0.0),
        (-sqrt3/2, -sqrt3, 0.0),        # Row 3: 2 hexes
        (sqrt3/2, -sqrt3, 0.0),
        (0.0, -2*sqrt3, 0.0),           # Row 4: 1 hex
    ]
    
    # Add 2 more hexagons with defects
    # Place one at top-left void, rotated 30°
    # Place one at bottom-right void, rotated 30°
    defect_positions = [
        (-1.5*sqrt3, 1.5*sqrt3, 30.0),   # Top-left, rotated
        (1.5*sqrt3, -1.5*sqrt3, 30.0),   # Bottom-right, rotated
    ]
    
    init = base_positions + defect_positions
    
    # Build initial parameter vector
    x0 = np.zeros(3*N_local)
    for i in range(N_local):
        x0[3*i] = init[i][0]
        x0[3*i+1] = init[i][1]
        x0[3*i+2] = init[i][2]
    
    # Define objective
    def objective(params):
        return compute_objective(params, N_local)
    
    # Try optimization with multiple restarts
    best_params = x0.copy()
    best_val = objective(x0)
    
    np.random.seed(42)
    
    # Multiple random restarts
    for restart in range(8):
        if restart > 0:
            # Perturb initial guess
            x0_perturbed = x0.copy()
            noise = np.random.normal(0, 0.3, size=3*N_local)
            x0_perturbed[:3*N_local] += noise
        else:
            x0_perturbed = x0
        
        try:
            result = minimize(
                objective, 
                x0_perturbed, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5}
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
    
    # Compute optimal outer hexagon
    min_side, outer_cx, outer_cy = compute_min_outer_side(inner, 0.0)
    
    # Ensure validity
    if min_side > 100:  # Fallback
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N_local // 2), N_local - N_local // 2)], (0.0, 0.0), 40.0, 0.0
    
    # Final validation
    outer_poly = hexagon_vertices(outer_cx, outer_cy, min_side, 0.0)
    inner_polys = []
    for i in range(N_local):
        poly = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        inner_polys.append(poly)
        if not check_inside(poly, outer_poly):
            return [[4.0 * k, 0.0, 0.0] for k in range(-(N_local // 2), N_local - N_local // 2)], (0.0, 0.0), 40.0, 0.0
    
    for i in range(N_local):
        for j in range(i+1, N_local):
            if polygons_intersect(inner_polys[i], inner_polys[j]):
                return [[4.0 * k, 0.0, 0.0] for k in range(-(N_local // 2), N_local - N_local // 2)], (0.0, 0.0), 40.0, 0.0
    
    return inner, (outer_cx, outer_cy), min_side, 0.0

# EVOLVE-BLOCK-START
# (construct function defined above)
# EVOLVE-BLOCK-END
