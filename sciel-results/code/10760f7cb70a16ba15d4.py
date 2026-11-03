import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0

def _hex_vertices(cx, cy, theta_deg):
    """Return 6 vertices of a unit regular hexagon."""
    theta = math.radians(theta_deg)
    verts = []
    for k in range(6):
        a = theta + math.radians(60 * k)
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _point_in_hex(px, py, side, angle_deg=0.0):
    """Check if point is inside a hexagon centered at (0,0) with given side and angle."""
    dx = px
    dy = py
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * SQRT3 / 2
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True

def _min_dist_point_segment(px, py, x1, y1, x2, y2):
    """Minimum distance from point (px,py) to segment (x1,y1)-(x2,y2)."""
    vx = x2 - x1
    vy = y2 - y1
    wx = px - x1
    wy = py - y1
    
    c1 = vx * wx + vy * wy
    if c1 <= 0:
        return math.hypot(px - x1, py - y1)
        
    c2 = vx * vx + vy * vy
    if c1 >= c2:
        return math.hypot(px - x2, py - y2)
        
    b = c1 / c2
    px_proj = x1 + b * vx
    py_proj = y1 + b * vy
    return math.hypot(px - px_proj, py - py_proj)

def _hex_distance(v1, v2):
    """Calculate minimum distance between two hexagons defined by their vertices.
    Returns 0 if they overlap or touch, positive if disjoint."""
    # Check for overlap: if any vertex of v1 is inside v2 or vice versa
    for vx, vy in v1:
        if _point_in_hex(vx, vy, 1.0, 0.0): # Wait, this checks against a hex at 0,0. 
            # We need to check against v2's local frame or transform v1 to v2's frame.
            pass
            
    # Simpler: Use Separating Axis Theorem (SAT) or just check vertex-edge distances if no overlap.
    # For unit hexagons, if they are disjoint, the min distance is the min of vertex-to-edge distances.
    # First, check if they are far apart.
    min_dist = float('inf')
    
    # To check overlap efficiently, we can check if any vertex of v1 is inside v2.
    # But v2 is not at 0,0.
    # Let's find the center of v1 and v2.
    c1x = sum(v[0] for v in v1) / 6.0
    c1y = sum(v[1] for v in v1) / 6.0
    c2x = sum(v[0] for v in v2) / 6.0
    c2y = sum(v[1] for v in v2) / 6.0
    
    # Approximate angle of v1 and v2? 
    # Actually, we can just check all vertex-edge pairs.
    # If min_dist > 0, they are disjoint.
    # If min_dist == 0, they touch or overlap.
    # To distinguish overlap from touch, we need to check if any vertex is strictly inside.
    
    # For the purpose of the penalty, if min_dist < eps, we consider it a violation.
    
    # Optimization: If centers are far apart, skip.
    if (c1x - c2x)**2 + (c1y - c2y)**2 > 4.0: # (1+1)^2
         return 1.0 # Arbitrary large positive distance if far apart? No, we need the actual distance for the constraint.
         # But for penalty, if distance > 0, penalty is 0.
         # So we can return a large number if we are sure they are disjoint.
         # However, we need to be sure.
         # Let's just compute the min distance.
         
    for vx, vy in v1:
        for i in range(6):
            x1, y1 = v2[i]
            x2, y2 = v2[(i+1)%6]
            d = _min_dist_point_segment(vx, vy, x1, y1, x2, y2)
            if d < min_dist:
                min_dist = d
                if min_dist < 1e-9:
                    # Potential overlap or touch. Need to check if strictly inside.
                    # If any vertex of v1 is strictly inside v2, distance is negative (overlap).
                    # For now, let's assume if min_dist is 0, it's a boundary case.
                    # We will treat min_dist < 0 as overlap.
                    # But _min_dist_point_segment returns >= 0.
                    # So we need an explicit overlap check.
                    pass
                    
    # Explicit overlap check
    # Check if any vertex of v1 is inside v2
    # We need a point_in_polygon function for v2.
    # Since v2 is a regular hexagon, we can use the center and angle.
    # But we only have vertices.
    # Let's derive center and angle from vertices.
    # This is getting complicated.
    
    # Alternative: Use the fact that for convex polygons, if they are disjoint, 
    # there is a separating axis.
    # For penalty, let's just use the min vertex-edge distance.
    # If they overlap, the min vertex-edge distance will be 0 (if a vertex is on an edge) 
    # or we might miss it if one polygon is fully inside the other (unlikely for unit hexagons unless identical).
    # If one is inside the other, min_dist will be > 0 (distance to boundary).
    # So we MUST check for containment.
    
    # Let's implement a simple point-in-convex-polygon test using cross products.
    def _point_in_convex_poly(px, py, poly):
        # poly is list of (x,y) in counter-clockwise order
        n = len(poly)
        if n < 3: return False
        # Check if point is on same side of all edges
        # For convex polygon, all cross products should have same sign
        sign = 0
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1)%n]
            cross = (x2-x1)*(py-y1) - (y2-y1)*(px-x1)
            if abs(cross) < 1e-9:
                continue
            if sign == 0:
                sign = 1 if cross > 0 else -1
            elif (cross > 0 and sign < 0) or (cross < 0 and sign > 0):
                return False
        return True

    # Check overlap
    for vx, vy in v1:
        if _point_in_convex_poly(vx, vy, v2):
            return -1.0 # Overlap
    for vx, vy in v2:
        if _point_in_convex_poly(vx, vy, v1):
            return -1.0 # Overlap
            
    return min_dist

def _calc_min_enclosing_side(vertices):
    """Calculate the side length of the minimal regular hexagon (centered at 0,0, angle 0)
    that contains all given vertices."""
    max_proj = 0.0
    # Normals for standard hexagon (flat top? pointy top?)
    # Standard: vertices at angles 0, 60, 120...
    # Normals at 30, 90, 150...
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        nx = math.cos(theta)
        ny = math.sin(theta)
        for vx, vy in vertices:
            proj = vx * nx + vy * ny
            if proj > max_proj:
                max_proj = proj
    return max_proj / APOTHEM

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Variables: [x1, y1, theta1, ..., x11, y11, theta11]
    # Bounds: 
    # The cluster should be within a reasonable range, say [-5, 5]
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (0.0, 360.0)] * N

    def objective(z):
        if time.time() > deadline:
            return float('inf')
            
        # Unpack
        hexes = []
        all_verts = []
        for i in range(N):
            x = z[i*3]
            y = z[i*3+1]
            theta = z[i*3+2]
            verts = _hex_vertices(x, y, theta)
            hexes.append(verts)
            all_verts.extend(verts)
            
        # Calculate S
        S = _calc_min_enclosing_side(all_verts)
        
        # Calculate Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = _hex_distance(hexes[i], hexes[j])
                if d < 0:
                    # Overlap
                    # Penalty proportional to depth of overlap? 
                    # d is -1.0 for overlap. 
                    # We want to push them apart.
                    # Let's use a large penalty.
                    penalty += 100.0
                elif d < 1e-6:
                    # Touching or very close
                    # We want strictly disjoint interiors, so touching is OK?
                    # "disjoint regular hexagons" usually implies interiors are disjoint.
                    # So d >= 0 is fine.
                    pass
                    
        return S + penalty

    # Initial guesses
    # 1. Lattice configuration
    sqrt3 = math.sqrt(3)
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)
        
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
             
    # Pick 4 from ring2 to make 11
    # Simple selection: first 4
    selected_r2 = ring2[:4]
    full_axial = pos_axial + selected_r2
    full_xy = [axial_to_xy(q, r) for q, r in full_axial]
    
    z0_lattice = []
    for x, y in full_xy:
        z0_lattice.extend([x, y, 0.0])
        
    best_score = float('inf')
    best_z = None
    
    # Try local optimization from lattice start
    try:
        res = minimize(objective, z0_lattice, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8})
        if res.fun < best_score and res.fun < 10: # Heuristic check for validity
            best_score = res.fun
            best_z = res.x
    except Exception as e:
        pass

    # Try random restarts if time permits
    np.random.seed(42)
    max_restarts = 10
    for k in range(max_restarts):
        if time.time() > deadline - 10:
            break
        z0_rand = []
        for i in range(N):
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-2, 2)
            theta = np.random.uniform(0, 360)
            z0_rand.extend([x, y, theta])
            
        try:
            res = minimize(objective, z0_rand, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-8})
            if res.fun < best_score and res.fun < 10:
                best_score = res.fun
                best_z = res.x
        except Exception as e:
            pass

    # If we found a valid solution, extract it
    if best_z is not None:
        inner = []
        for i in range(N):
            x = best_z[i*3]
            y = best_z[i*3+1]
            theta = best_z[i*3+2]
            inner.append([x, y, theta])
        
        # Recalculate S for the final answer
        all_verts = []
        for i in range(N):
            x = best_z[i*3]
            y = best_z[i*3+1]
            theta = best_z[i*3+2]
            verts = _hex_vertices(x, y, theta)
            all_verts.extend(verts)
            
        S = _calc_min_enclosing_side(all_verts)
        
        # The outer hexagon is centered at (0,0) with angle 0
        return inner, (0.0, 0.0), S, 0.0

    # Fallback to lattice if optimization failed
    inner = [[x, y, 0.0] for x, y in full_xy]
    all_verts = []
    for x, y in full_xy:
        verts = _hex_vertices(x, y, 0.0)
        all_verts.extend(verts)
    S = _calc_min_enclosing_side(all_verts)
    return inner, (0.0, 0.0), S, 0.0

