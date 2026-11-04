import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 12

def get_hex_verts(cx, cy, side, rot):
    """Return vertices of a regular hexagon as a (6, 2) array."""
    angles = np.arange(6) * np.pi / 3 + rot
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def point_in_poly(pt, poly):
    """Check if point is inside convex polygon (CCW)."""
    n = len(poly)
    sign = 0
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        v1 = p2 - p1
        v2 = pt - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross > 1e-9:
            if sign == -1:
                return False
            sign = 1
        elif cross < -1e-9:
            if sign == 1:
                return False
            sign = -1
    return True

def dist_point_to_segment(pt, p1, p2):
    """Distance from point to line segment p1-p2."""
    v = p2 - p1
    w = pt - p1
    c1 = np.dot(w, v)
    if c1 <= 0:
        return np.linalg.norm(w)
    c2 = np.dot(v, v)
    if c2 <= c1:
        return np.linalg.norm(pt - p2)
    b = c1 / c2
    v_b = b * v
    return np.linalg.norm(w - v_b)

def dist_point_to_poly(pt, poly):
    """Signed distance from point to convex polygon. Negative if inside."""
    inside = point_in_poly(pt, poly)
    min_dist = np.inf
    n = len(poly)
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        d = dist_point_to_segment(pt, p1, p2)
        if d < min_dist:
            min_dist = d
    if inside:
        return -min_dist
    else:
        return min_dist

def dist_poly_to_poly(poly1, poly2):
    """Min signed distance between two convex polygons. Negative if overlap."""
    min_d = np.inf
    # Check vertices of poly1 against poly2
    for v in poly1:
        d = dist_point_to_poly(v, poly2)
        if d < min_d:
            min_d = d
    # Check vertices of poly2 against poly1
    for v in poly2:
        d = dist_point_to_poly(v, poly1)
        if d < min_d:
            min_d = d
    return min_d

def cost(params):
    S = params[0]
    phi = params[1]
    inner_params = params[2:]
    
    outer_verts = get_hex_verts(0.0, 0.0, S, phi)
    
    total_penalty = 0.0
    hex_verts_list = []
    
    for i in range(N):
        x, y, theta = inner_params[i*3], inner_params[i*3+1], inner_params[i*3+2]
        verts = get_hex_verts(x, y, 1.0, theta)
        hex_verts_list.append(verts)
        
        # Containment penalty
        for v in verts:
            d = dist_point_to_poly(v, outer_verts)
            if d < 0:
                total_penalty += d * d
                
    # Overlap penalty
    for i in range(N):
        for j in range(i+1, N):
            d = dist_poly_to_poly(hex_verts_list[i], hex_verts_list[j])
            if d < 0:
                total_penalty += d * d * 10.0
    
    return S + total_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    bounds = [(3.5, 6.0), (0, np.pi/3)]
    for _ in range(N):
        bounds.append((-5, 5))
        bounds.append((-5, 5))
        bounds.append((0, np.pi/3))
        
    best_params = None
    best_cost = np.inf
    
    try:
        res = differential_evolution(
            cost, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=1e-4,
            seed=42,
            polish=False
        )
        best_params = res.x
        best_cost = res.fun
    except Exception:
        pass
        
    if best_params is not None and time.time() < deadline - 10:
        try:
            res2 = minimize(
                cost, 
                best_params, 
                method='Powell', 
                options={'maxiter': 100, 'disp': False}
            )
            if res2.fun < best_cost:
                best_params = res2.x
                best_cost = res2.fun
        except Exception:
            pass
            
    S = best_params[0]
    phi = best_params[1]
    inner = []
    for i in range(N):
        x = best_params[2 + i*3]
        y = best_params[3 + i*3]
        theta = best_params[4 + i*3]
        inner.append([x, y, np.degrees(theta)])
        
    center = (0.0, 0.0)
    angle_deg = np.degrees(phi)
    
    return inner, center, S, angle_deg
