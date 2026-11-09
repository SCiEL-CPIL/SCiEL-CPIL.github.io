"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_DEADLINE = 145.0

def _hex_inradius(s):
    return s * math.cos(math.radians(30))

def _point_in_hex(px, py, cx, cy, s, rot):
    """
    Check if point (px, py) is inside a hexagon centered at (cx, cy) with side s and rotation rot.
    Returns True if inside.
    """
    dx = px - cx
    dy = py - cy
    ca, sa = math.cos(-rot), math.sin(-rot)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    r_in = _hex_inradius(s)
    c30 = math.cos(math.radians(30))
    # Hexagon with flat top/bottom (rot=0) has vertices at 0, 60, 120...
    # Wait, standard "pointy top" vs "flat top".
    # If vertices are at 0, 60, 120, 180, 240, 300:
    # The sides are at 30, 90, 150, 210, 270, 330.
    # Distance to side with normal (cos(30), sin(30)) is |lx cos(30) + ly sin(30)|.
    # Distance to side with normal (cos(90), sin(90)) is |ly|.
    # Distance to side with normal (cos(150), sin(150)) is |lx cos(150) + ly sin(150)|.
    d1 = abs(c30 * lx + 0.5 * ly)
    d2 = abs(ly)
    d3 = abs(-c30 * lx + 0.5 * ly)
    return max(d1, d2, d3) <= r_in + 1e-9

def _hex_verts(cx, cy, s, rot):
    vs = np.empty((6, 2))
    for k in range(6):
        a = math.radians(60.0 * k) + rot
        vs[k, 0] = cx + s * math.cos(a)
        vs[k, 1] = cy + s * math.sin(a)
    return vs

def _min_dist_poly(A, B):
    """Compute minimum distance between two convex polygons (lists of 2D points).
    Returns 0.0 if they overlap, otherwise the minimum edge-to-edge distance.
    Uses SAT (Separating Axis Theorem) approach for convex polygons."""
    nA = len(A)
    nB = len(B)
    
    # Check if any vertex of A is inside B or vice versa (overlap)
    def point_in_poly(pt, poly):
        # Winding number test for convex polygon
        sign = 0
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % len(poly)]
            cross = (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)
            if cross > 1e-10:
                sign = 1
            elif cross < -1e-10:
                sign = -1
            if sign != 0:
                # All crosses should have same sign for point inside convex poly
                pass
        # Better: use half-plane test
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % len(poly)]
            # Normal pointing inward (for CCW polygon)
            nx = -(y2 - y1)
            ny = (x2 - x1)
            d = nx * (pt[0] - x1) + ny * (pt[1] - y1)
            if d < -1e-9:
                return False
        return True
    
    # Check vertex containment
    for v in A:
        if point_in_poly(v, B):
            return 0.0
    for v in B:
        if point_in_poly(v, A):
            return 0.0
    
    # No overlap: compute minimum edge-to-edge distance
    min_d = float('inf')
    for i in range(nA):
        ax1, ay1 = A[i]
        ax2, ay2 = A[(i+1) % nA]
        for j in range(nB):
            bx1, by1 = B[j]
            bx2, by2 = B[(j+1) % nB]
            # Distance from point to line segment
            def pt_seg_dist(px, py, x1, y1, x2, y2):
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    return math.sqrt((px-x1)**2 + (py-y1)**2)
                t = ((px-x1)*dx + (py-y1)*dy) / (dx*dx + dy*dy)
                t = max(0.0, min(1.0, t))
                qx = x1 + t*dx
                qy = y1 + t*dy
                return math.sqrt((px-qx)**2 + (py-qy)**2)
            
            d1 = pt_seg_dist(ax1, ay1, bx1, by1, bx2, by2)
            d2 = pt_seg_dist(ax2, ay2, bx1, by1, bx2, by2)
            d3 = pt_seg_dist(bx1, by1, ax1, ay1, ax2, ay2)
            d4 = pt_seg_dist(bx2, by2, ax1, ay1, ax2, ay2)
            d = min(d1, d2, d3, d4)
            if d < min_d:
                min_d = d
    
    return min_d

def _is_valid_config(inner, center, side, angle):
    """
    Strictly validate the configuration.
    inner: list of (x, y, angle_deg)
    center: (cx, cy)
    side: outer side length
    angle: outer rotation (radians)
    """
    cx, cy = center
    s_out = side
    r_out = _hex_inradius(s_out)
    
    # Check boundary
    for i in range(len(inner)):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        verts = _hex_verts(xi, yi, 1.0, ai)
        for v in verts:
            if not _point_in_hex(v[0], v[1], cx, cy, s_out, angle):
                return False

    # Check overlap
    for i in range(len(inner)):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        vi = _hex_verts(xi, yi, 1.0, ai)
        for j in range(i + 1, len(inner)):
            xj, yj, aj_deg = inner[j]
            aj = math.radians(aj_deg)
            vj = _hex_verts(xj, yj, 1.0, aj)
            
            # Quick reject
            dc = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dc > 2.0:
                continue
            
            # Check if any vertex of i is in j
            overlap = False
            for v in vi:
                if _point_in_hex(v[0], v[1], xj, yj, 1.0, aj):
                    overlap = True
                    break
            if overlap:
                return False
            
            # Check if any vertex of j is in i
            for v in vj:
                if _point_in_hex(v[0], v[1], xi, yi, 1.0, ai):
                    overlap = True
                    break
            if overlap:
                return False
                
    return True

def _honeycomb_positions():
    """Generate honeycomb packing positions for 11 hexagons."""
    positions = []
    # Center
    positions.append((0.0, 0.0, 0.0))
    # Ring 1: 6 hexagons at distance 2
    for k in range(6):
        a = math.radians(60.0 * k + 30.0)
        positions.append((2.0*math.cos(a), 2.0*math.sin(a), 0.0))
    # Ring 2: 4 hexagons at distance 4, alternating angles
    # Angles: 30, 90, 150, 210 (to match the gaps? No, to align with Ring 1)
    # Ring 1 is at 30, 90, 150, 210, 270, 330.
    # We pick 4 of them to extend.
    for k in range(4):
        a = math.radians(60.0 * k + 30.0)
        positions.append((4.0*math.cos(a), 4.0*math.sin(a), 0.0))
    return positions[:N]

def construct():
    t_start = time.time()
    n = N

    def _remaining():
        return _DEADLINE - (time.time() - t_start)

    # Generate initial valid configuration
    init_pos = _honeycomb_positions()
    
    # Calculate required outer side for initial config
    # Find max distance from origin to any vertex
    max_r = 0.0
    for (xi, yi, ai) in init_pos:
        verts = _hex_verts(xi, yi, 1.0, ai)
        for v in verts:
            r = math.sqrt(v[0]**2 + v[1]**2)
            if r > max_r:
                max_r = r
    
    # Outer hex inradius R_out must be >= max_r
    # R_out = S * cos(30)
    s_init = max_r / math.cos(math.radians(30))
    
    # Add small margin
    s_init *= 1.01
    
    # Prepare initial parameters for optimization
    # p = [x1, y1, a1, x2, y2, a2, ..., xn, yn, an, S]
    p0 = np.zeros(3*n + 1)
    for i in range(n):
        p0[3*i] = init_pos[i][0]
        p0[3*i+1] = init_pos[i][1]
        p0[3*i+2] = 0.0 # angle
    p0[3*n] = s_init
    
    def objective(p):
        x = p[0:3*n:3]
        y = p[1:3*n:3]
        angles = p[2:3*n:3]
        S = p[3*n]
        
        if S < 1.0:
            return 1e6
            
        pen = 0.0
        r_out = _hex_inradius(S)
        c30 = math.cos(math.radians(30))
        
        # Boundary penalty: smooth violation using max of (distance - r_out)
        for i in range(n):
            verts = _hex_verts(x[i], y[i], 1.0, angles[i])
            for v in verts:
                dx = v[0]
                dy = v[1]
                d1 = abs(c30 * dx + 0.5 * dy)
                d2 = abs(dy)
                d3 = abs(-c30 * dx + 0.5 * dy)
                d_max = max(d1, d2, d3)
                violation = max(0.0, d_max - r_out)
                pen += violation * violation
                    
        # Overlap penalty: use smooth minimum distance between hexagons
        for i in range(n):
            for j in range(i+1, n):
                dc = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dc > 2.0:
                    continue
                vi = _hex_verts(x[i], y[i], 1.0, angles[i]).tolist()
                vj = _hex_verts(x[j], y[j], 1.0, angles[j]).tolist()
                
                dist = _min_dist_poly(vi, vj)
                # If dist is 0, they overlap; apply penalty
                if dist < 1e-6:
                    pen += 5.0
                else:
                    # Small gap penalty to encourage separation
                    if dist < 0.1:
                        pen += (0.1 - dist) * 2.0
        
        return 0.5 * S * S + 10.0 * pen

    best_val = float('inf')
    best_p = p0.copy()
    
    # Try optimization with multiple strategies
    rng = np.random.RandomState(42)
    
    # First: Nelder-Mead from initial
    if _remaining() > 30.0:
        try:
            res = minimize(objective, p0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    # Second: Perturbed restarts
    for trial in range(4):
        if _remaining() < 25.0:
            break
            
        p_init = best_p.copy()
        noise = 0.15
        p_init[0:3*n] += rng.randn(3*n) * noise
        p_init[3*n] = max(1.0, s_init * (0.95 + rng.rand()*0.1))
        
        try:
            res = minimize(objective, p_init, method='Nelder-Mead',
                          options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    # Third: Polish pass with tighter tolerance
    if _remaining() > 15.0:
        try:
            res = minimize(objective, best_p, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x.copy()
        except Exception:
            pass

    # Extract best solution
    x = best_p[0:3*n:3]
    y = best_p[1:3*n:3]
    angles = best_p[2:3*n:3]
    S = best_p[3*n]
    
    inner = [[float(x[i]), float(y[i]), float(math.degrees(angles[i]))] for i in range(n)]
    center = (0.0, 0.0)
    angle_out = 0.0
    
    # Validate with margin
    if _is_valid_config(inner, center, S, angle_out):
        return inner, center, float(S), angle_out
    else:
        # Try to repair: slightly increase S
        for margin in [0.01, 0.02, 0.05, 0.1, 0.2]:
            S_try = S + margin
            if _is_valid_config(inner, center, S_try, angle_out):
                return inner, center, float(S_try), angle_out
        
        # Fallback to initial valid configuration
        inner_init = [[float(p[0]), float(p[1]), float(math.degrees(p[2]))] for p in init_pos]
        return inner_init, center, float(s_init), angle_out
