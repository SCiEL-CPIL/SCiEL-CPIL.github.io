import math
import random
from scipy.optimize import minimize

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side length 'side'."""
    verts = []
    for i in range(6):
        a = math.radians(angle_deg) + math.radians(60 * i)
        x = cx + side * math.cos(a)
        y = cy + side * math.sin(a)
        verts.append((x, y))
    return verts

def point_in_polygon(x, y, polygon):
    """Ray casting algorithm to check if point (x,y) is inside polygon."""
    n = len(polygon)
    inside = False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside

def hexagon_contains_point(polygon, px, py):
    """Check if point (px, py) is inside the hexagon defined by polygon vertices."""
    return point_in_polygon(px, py, polygon)

def polygons_disjoint(poly1, poly2, eps=1e-9):
    """Check if two polygons are disjoint (no interior overlap)."""
    # Check if any vertex of poly1 is inside poly2 or vice versa
    for p in poly1:
        if point_in_polygon(p[0], p[1], poly2):
            return False
    for p in poly2:
        if point_in_polygon(p[0], p[1], poly1):
            return False
    return True

def all_inner_disjoint(inner_hexes):
    """Check if all inner hexagons are pairwise disjoint."""
    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if not polygons_disjoint(inner_hexes[i], inner_hexes[j]):
                return False
    return True

def all_inside_outer(inner_hexes, outer_poly):
    """Check if all vertices of all inner hexagons are inside the outer hexagon."""
    for inner in inner_hexes:
        for v in inner:
            if not point_in_polygon(v[0], v[1], outer_poly):
                return False
    return True

def objective(params):
    """Objective function: outer side length + penalty for violations."""
    n = len(params) // 3
    inner = []
    for i in range(n):
        x, y, ang = params[3*i], params[3*i+1], math.degrees(params[3*i+2])
        inner.append(hex_vertices(x, y, 1.0, ang))
    
    # Estimate required outer side length: find max distance from center to any vertex
    max_dist = 0
    for inner in inner:
        for v in inner:
            d = math.sqrt(v[0]**2 + v[1]**2)
            if d > max_dist:
                max_dist = d
    
    # For a regular hexagon, side length s has inradius s*sqrt(3)/2
    # So to contain a point at distance d from center, we need inradius >= d
    # s >= d / (sqrt(3)/2) = 2*d/sqrt(3)
    outer_side = 2.0 * max_dist / math.sqrt(3)
    
    # Create outer hexagon and check if all inner hexagons fit
    outer_poly = hex_vertices(0, 0, outer_side, 0)
    
    penalty = 0.0
    if not all_inside_outer(inner, outer_poly):
        penalty += 1000.0
    if not all_inner_disjoint(inner):
        penalty += 1000.0
    
    return outer_side + penalty

def construct():
    """Construct 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
    # Initial guess: place hexagons in a hexagonal pattern
    # Center at origin, with 1 center hex and 6 around, plus more
    # Use a honeycomb-like arrangement
    
    # Let's try a symmetric arrangement
    # Place hexagons at positions that form a cluster
    
    # Initial positions: center + 6 neighbors + 5 more
    # For side length 1, distance between centers of adjacent hexes is sqrt(3)
    d = math.sqrt(3)
    
    initial_positions = []
    # Center
    initial_positions.append((0, 0, 0))
    # 6 neighbors
    for i in range(6):
        angle = math.radians(60 * i)
        initial_positions.append((d * math.cos(angle), d * math.sin(angle), 0))
    # 5 more at distance 2d from center
    for i in range(5):
        angle = math.radians(60 * i + 30)
        initial_positions.append((2*d * math.cos(angle), 2*d * math.sin(angle), 0))
    
    x0 = []
    for x, y, ang_deg in initial_positions:
        x0.extend([x, y, math.radians(ang_deg)])
    
    # Optimize
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
    
    # Extract solution
    params = result.x
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        angle_deg = math.degrees(params[3*i+2])
        inner.append([x, y, angle_deg])
    
    # Calculate outer side length
    max_dist = 0
    for i in range(N):
        x, y, ang_deg = inner[i]
        verts = hex_vertices(x, y, 1.0, ang_deg)
        for v in verts:
            d = math.sqrt(v[0]**2 + v[1]**2)
            if d > max_dist:
                max_dist = d
    
    outer_side = 2.0 * max_dist / math.sqrt(3)
    
    return inner, (0.0, 0.0), outer_side, 0.0

# EVOLVE-BLOCK-START
# (construct function defined above)
# EVOLVE-BLOCK-END
