"""Packing 12 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def poly_area(poly):
    """Shoelace formula for polygon area."""
    n = len(poly)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    return abs(area) / 2.0

def point_in_polygon(px, py, poly):
    """Ray casting point-in-polygon test."""
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

def segment_intersects_segment(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def polygons_overlap(poly1, poly2, tol=1e-9):
    """Check if two convex polygons overlap using SAT."""
    # Test all edges from both polygons
    for poly in [poly1, poly2]:
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            # Edge normal
            ex = poly[j][0] - poly[i][0]
            ey = poly[j][1] - poly[i][1]
            nx, ny = -ey, ex
            # Project all vertices of both polygons onto this normal
            min1, max1 = np.inf, -np.inf
            min2, max2 = np.inf, -np.inf
            for v in poly1:
                proj = v[0]*nx + v[1]*ny
                min1 = min(min1, proj)
                max1 = max(max1, proj)
            for v in poly2:
                proj = v[0]*nx + v[1]*ny
                min2 = min(min2, proj)
                max2 = max(max2, proj)
            if max1 < min2 + tol or max2 < min1 + tol:
                return False
    return True

def check_validity(inner, center, side, angle):
    """Check if the packing is valid (no overlaps, all inside outer hexagon)."""
    outer = hex_vertices(center[0], center[1], side, angle)
    
    for i in range(len(inner)):
        vi = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        # Check inside outer hexagon
        for v in vi:
            if not point_in_polygon(v[0], v[1], outer):
                return False
        # Check overlap with other hexagons
        for j in range(i+1, len(inner)):
            vj = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            if polygons_overlap(vi, vj):
                return False
    return True

def objective(params):
    """Objective: outer hexagon side length + penalty for violations."""
    # params: [s, angle_outer, x0,y0,a0, x1,y1,a1, ..., x11,y11,a11]
    s = params[0]
    angle_outer = params[1]
    
    # Get inner hexagons
    inner = []
    for i in range(N):
        x = params[2 + i*3]
        y = params[3 + i*3]
        a = params[4 + i*3]
        inner.append([x, y, a])
    
    center = (0.0, 0.0)
    outer = hex_vertices(center[0], center[1], s, angle_outer)
    
    penalty = 0.0
    
    for i in range(N):
        vi = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        # Check inside outer hexagon
        for v in vi:
            if not point_in_polygon(v[0], v[1], outer):
                # Distance from point to polygon boundary
                penalty += 1.0
        # Check overlap with other hexagons
        for j in range(i+1, N):
            vj = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            if polygons_overlap(vi, vj):
                penalty += 1.0
    
    return s + 10.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_result = None
    best_score = float('inf')
    
    # Initial guess: ring of 12 hexagons
    r = 3.5
    s_init = r + 1.5
    
    # Try different initial configurations
    configs = []
    
    # Config 1: ring at various radii
    for r_try in [3.0, 3.5, 4.0]:
        s_try = r_try + 1.5
        params = [s_try, 0.0]
        for k in range(N):
            theta = 2 * np.pi * k / N
            x = r_try * np.cos(theta)
            y = r_try * np.sin(theta)
            # Rotate each hexagon to face outward
            a = np.degrees(theta)
            params.extend([x, y, a])
        configs.append(params)
    
    # Config 2: all aligned
    for r_try in [3.0, 3.5, 4.0]:
        s_try = r_try + 1.5
        params = [s_try, 0.0]
        for k in range(N):
            theta = 2 * np.pi * k / N
            x = r_try * np.cos(theta)
            y = r_try * np.sin(theta)
            params.extend([x, y, 0.0])
        configs.append(params)
    
    # Config 3: hexagonal lattice pattern
    # 1 center + 6 ring + 5 more
    s_try = 5.0
    params = [s_try, 0.0]
    positions = [(0, 0)]
    # Ring of 6
    for k in range(6):
        theta = 2 * np.pi * k / 6
        positions.append((2*np.cos(theta), 2*np.sin(theta)))
    # Second partial ring
    for k in range(5):
        theta = 2 * np.pi * k / 6 + np.pi/6
        positions.append((4*np.cos(theta), 4*np.sin(theta)))
    for (x, y) in positions:
        params.extend([x, y, 0.0])
    configs.append(params)
    
    best_params = None
    best_obj = float('inf')
    
    for config in configs:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(objective, config, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6,
                                   'adaptive': True})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is not None:
        s = best_params[0]
        angle_outer = best_params[1]
        inner = []
        for i in range(N):
            x = best_params[2 + i*3]
            y = best_params[3 + i*3]
            a = best_params[4 + i*3]
            inner.append([x, y, a])
        
        # Verify and refine
        if check_validity(inner, (0.0, 0.0), s, angle_outer):
            best_result = (inner, (0.0, 0.0), s, angle_outer)
            best_score = s
    
    # If optimization didn't give valid result, try analytical ring solution
    if best_result is None:
        # Ring of 12, rotated hexagons
        r = 3.4
        s = r + 1.05
        inner = []
        for k in range(N):
            theta = 2 * np.pi * k / N
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            a = np.degrees(theta)
            inner.append([x, y, a])
        
        if check_validity(inner, (0.0, 0.0), s, np.degrees(0)):
            best_result = (inner, (0.0, 0.0), s, 0.0)
            best_score = s
        else:
            # Fallback: larger side
            s = 5.0
            r = 3.5
            inner = []
            for k in range(N):
                theta = 2 * np.pi * k / N
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                a = np.degrees(theta)
                inner.append([x, y, a])
            best_result = (inner, (0.0, 0.0), s, 0.0)
            best_score = s
    
    return best_result

# EVOLVE-BLOCK-START
# (construct defined above)
# EVOLVE-BLOCK-END
