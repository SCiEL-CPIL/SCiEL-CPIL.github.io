"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import math

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_deg):
    """Return the 6 vertices of a regular hexagon."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * math.pi / 3
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return np.array(verts)

def _hex_vertices_unit(cx, cy, angle_deg):
    """Unit side length hexagon vertices."""
    return _hex_vertices(cx, cy, 1.0, angle_deg)

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and angle."""
    dx = px - cx
    dy = py - cy
    a = math.radians(angle_deg)
    ca, sa = math.cos(-a), math.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    apothem = side * math.sqrt(3) / 2
    for i in range(6):
        na = math.radians(30 + i * 60)
        nx = math.cos(na)
        ny = math.sin(na)
        if nx * lx + ny * ly > apothem + 1e-12:
            return False
    return True

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _hexagons_overlap(h1, h2):
    """
    Check if two unit hexagons overlap.
    h1, h2 are (cx, cy, angle_deg).
    Returns True if they overlap (interiors intersect or boundaries touch significantly).
    """
    cx1, cy1, a1 = h1
    cx2, cy2, a2 = h2
    
    # Quick bounding box check
    dc = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    if dc > 4.0:
        return False
    
    verts1 = _hex_vertices_unit(cx1, cy1, a1)
    verts2 = _hex_vertices_unit(cx2, cy2, a2)
    
    # Check if any vertex of h1 is inside h2
    for v in verts1:
        if _point_in_hex(v[0], v[1], cx2, cy2, 1.0, a2):
            return True
            
    # Check if any vertex of h2 is inside h1
    for v in verts2:
        if _point_in_hex(v[0], v[1], cx1, cy1, 1.0, a1):
            return True
            
    # Check edge intersections
    for i in range(6):
        p1 = verts1[i]
        p2 = verts1[(i + 1) % 6]
        for j in range(6):
            q1 = verts2[j]
            q2 = verts2[(j + 1) % 6]
            if _segments_intersect(p1, p2, q1, q2):
                return True
                
    return False

def _outer_side_length(inner, outer_center, outer_angle):
    """
    Compute the minimum outer hexagon side length that contains all inner hexagons.
    inner: list of (cx, cy, angle_deg)
    outer_center: (cx, cy)
    outer_angle: degrees
    """
    ox, oy = outer_center
    oa = outer_angle
    
    max_extent = 0.0
    for (cx, cy, ia) in inner:
        verts = _hex_vertices_unit(cx, cy, ia)
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            a = math.radians(oa)
            ca, sa = math.cos(-a), math.sin(-a)
            lx = ca * dx - sa * dy
            ly = sa * dx + ca * dy
            
            for k in range(6):
                na = math.radians(30 + k * 60)
                nx = math.cos(na)
                ny = math.sin(na)
                proj = nx * lx + ny * ly
                if proj > max_extent:
                    max_extent = proj
    
    S = max_extent * 2.0 / math.sqrt(3)
    return max(S, 1.0)

def _objective(params, outer_center, outer_angle):
    """
    Objective: minimize outer side length + penalty for overlaps.
    params: array of [cx1, cy1, a1, cx2, cy2, a2, ...] for N inner hexagons
    """
    n = len(params) // 3
    inner = []
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        a = params[3*i+2]
        inner.append((cx, cy, a))
    
    # Check for overlaps
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            if _hexagons_overlap(inner[i], inner[j]):
                # Strong penalty
                cx1, cy1, _ = inner[i]
                cx2, cy2, _ = inner[j]
                dist = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                # The more they overlap, the higher the penalty
                # If dist < 2*sqrt(3) approx, they are likely overlapping
                # Use a quadratic penalty based on how much they violate the minimum distance
                # Minimum distance between centers for non-overlapping hexagons is sqrt(3)
                # But depending on rotation, it can be less.
                # Simple heuristic: penalty proportional to (sqrt(3) - dist)^2 if dist < sqrt(3)
                # But this is not exact. Let's just use a large constant penalty for any overlap.
                penalty += 10.0
                
    S = _outer_side_length(inner, outer_center, outer_angle)
    return S + penalty

def _repair_overlaps(inner, max_iter=100):
    """
    Iteratively push overlapping hexagons apart.
    inner: list of [cx, cy, angle]
    Returns modified list.
    """
    inner = [list(h) for h in inner]
    n = len(inner)
    
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                if _hexagons_overlap(tuple(inner[i]), tuple(inner[j])):
                    # Push apart
                    cx1, cy1, a1 = inner[i]
                    cx2, cy2, a2 = inner[j]
                    dx = cx2 - cx1
                    dy = cy2 - cy1
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 1e-6:
                        # Random nudge
                        dx = np.random.randn()
                        dy = np.random.randn()
                        dist = math.sqrt(dx*dx + dy*dy)
                    
                    # Push each by half the required separation
                    # Approximate required separation is sqrt(3)
                    # We just push them apart by a small amount along the axis
                    push = 0.1
                    inner[i][0] -= (dx / dist) * push
                    inner[i][1] -= (dy / dist) * push
                    inner[j][0] += (dx / dist) * push
                    inner[j][1] += (dy / dist) * push
                    moved = True
        if not moved:
            break
            
    return inner

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration: hexagonal packing
    # For N=11, use a 1-2-3-2-1 pattern or similar
    # Let's use a compact hexagonal arrangement
    
    def initial_config(n):
        config = []
        # Center
        config.append([0.0, 0.0, 0.0])
        # 6 neighbors at distance sqrt(3) in directions 0, 60, 120, 180, 240, 300
        for i in range(6):
            a = i * math.pi / 3
            x = math.sqrt(3) * math.cos(a)
            y = math.sqrt(3) * math.sin(a)
            config.append([x, y, 0.0])
        # 4 more: next ring
        # Positions at distance 2*sqrt(3) in directions 30, 90, 150, 210, 270, 330
        # Or at distance sqrt(3)*sqrt(3)=3 in directions 30, 90, ...
        # Let's just add 4 more at reasonable positions to complete the 11
        # The next ring of a hexagonal lattice has 12 positions.
        # We need 4 more.
        # Let's place them at the "corners" of the second ring
        # Angles: 0, 60, 120, 180, 240, 300 for the first ring (distance sqrt(3))
        # Second ring positions:
        # Distance 2*sqrt(3) at angles 0, 60, ...
        # Distance sqrt(3)*sqrt(3) = 3 at angles 30, 90, ...
        
        extra_positions = [
            (2*math.sqrt(3), 0),
            (math.sqrt(3), 3),
            (-math.sqrt(3), 3),
            (-2*math.sqrt(3), 0),
        ]
        for i in range(min(4, n - 7)):
            x, y = extra_positions[i]
            config.append([x, y, 0.0])
            
        return config[:n]
    
    init = initial_config(N)
    
    # Parameters: [cx1, cy1, a1, cx2, cy2, a2, ..., cxN, cyN, aN]
    init_params = np.array([v for hex in init for v in hex])
    
    def obj_func(params):
        return _objective(params, (0.0, 0.0), 0.0)
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))  # cx
        bounds.append((-10, 10))  # cy
        bounds.append((-180, 180))  # angle
    
    best_S = float('inf')
    best_params = init_params.copy()
    
    # Use differential evolution for global optimization
    if time.time() < deadline - 30:
        try:
            result = differential_evolution(
                obj_func, bounds, 
                maxiter=100, popsize=15,
                tol=1e-4, seed=42,
                polish=True,
                init='sobol'
            )
            if result.fun < best_S:
                best_S = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Local polish with L-BFGS-B
    if time.time() < deadline - 10:
        try:
            result = minimize(
                obj_func, best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            if result.fun < best_S:
                best_S = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Extract final configuration
    inner = []
    for i in range(N):
        cx = best_params[3*i]
        cy = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([cx, cy, a])
    
    # Repair overlaps
    inner = _repair_overlaps(inner)
    
    # Recompute outer side length after repair
    outer_side = _outer_side_length(inner, (0.0, 0.0), 0.0)
    
    return inner, (0.0, 0.0), outer_side, 0.0

# EVOLVE-BLOCK-END
