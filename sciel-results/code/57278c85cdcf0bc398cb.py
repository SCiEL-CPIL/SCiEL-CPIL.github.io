"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def get_normal(p1, p2):
    """Get normal vector for edge p1-p2."""
    edge = p2 - p1
    return np.array([-edge[1], edge[0]])

def project_points(points, axis):
    return np.dot(points, axis)

def polygon_penetration_depth(v1, v2, tol=1e-9):
    """
    Returns the minimum penetration depth if polygons overlap, 0 otherwise.
    Uses SAT.
    """
    polys = [v1, v2]
    min_overlap = float('inf')
    
    for poly in polys:
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            normal = get_normal(poly[i], poly[j])
            norm = np.linalg.norm(normal)
            if norm < 1e-9: continue
            normal = normal / norm
            
            proj1 = project_points(v1, normal)
            proj2 = project_points(v2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Overlap amount
            overlap = min(max1, max2) - max(min1, min2)
            
            if overlap < -tol:
                return 0.0 # Separated
            
            min_overlap = min(min_overlap, overlap)
            
    return min_overlap

def point_in_hexagon(p, cx, cy, side, angle_deg):
    """Check if point p is inside regular hexagon centered at (cx,cy)."""
    dx = p[0] - cx
    dy = p[1] - cy
    a = -math.radians(angle_deg)
    rx = dx * math.cos(a) - dy * math.sin(a)
    ry = dx * math.sin(a) + dy * math.cos(a)
    
    apothem = side * math.sqrt(3) / 2.0
    
    for k in range(6):
        theta = k * math.pi / 3.0
        nx = math.cos(theta)
        ny = math.sin(theta)
        dist = abs(rx * nx + ry * ny)
        if dist > apothem + 1e-9:
            return False
    return True

def energy_function(params, S, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Calculate penalty energy.
    params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12]
    S: outer side length
    """
    energy = 0.0
    verts_list = []
    
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        verts = hex_vertices(x, y, a, 1.0)
        verts_list.append(verts)
        
        # Constraint 1: Inside outer hexagon
        for v in verts:
            if not point_in_hexagon(v, outer_center[0], outer_center[1], S, outer_angle):
                dx = v[0] - outer_center[0]
                dy = v[1] - outer_center[1]
                ra = -math.radians(outer_angle)
                rx = dx * math.cos(ra) - dy * math.sin(ra)
                ry = dx * math.sin(ra) + dy * math.cos(ra)
                
                apothem = S * math.sqrt(3) / 2.0
                min_dist = float('inf')
                for k in range(6):
                    theta = k * math.pi / 3.0
                    nx = math.cos(theta)
                    ny = math.sin(theta)
                    dist = abs(rx * nx + ry * ny)
                    pen = max(0.0, dist - apothem)
                    min_dist = min(min_dist, pen)
                energy += 10.0 * min_dist * min_dist

    # Constraint 2: No overlap
    for i in range(N):
        for j in range(i+1, N):
            pen = polygon_penetration_depth(verts_list[i], verts_list[j])
            if pen > 1e-9:
                energy += 100.0 * pen * pen

    return energy

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    # Initial guess: Hexagonal lattice
    d = 2.0 # Distance between centers for touching hexes
    initial_params = []
    positions = []
    positions.append((0,0))
    for k in range(6):
        angle = k * math.pi / 3
        positions.append((d*math.cos(angle), d*math.sin(angle)))
    for k in range(5):
        angle = k * 2 * math.pi / 5
        r = d * 2.5
        positions.append((r*math.cos(angle), r*math.sin(angle)))
        
    for x, y in positions:
        initial_params.extend([x, y, 0.0])
        
    initial_params = np.array(initial_params)

    best_S = 10.0
    best_config = initial_params
    
    lo = 3.0
    hi = 5.0
    
    while lo < hi - 1e-4 and time.time() < deadline:
        mid = (lo + hi) / 2.0
        
        feasible = False
        best_local_config = None
        
        # Use Nelder-Mead
        try:
            res = minimize(energy_function, initial_params, args=(mid,), method='Nelder-Mead', 
                       options={'maxiter': 800, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < 1e-3:
                feasible = True
                best_local_config = res.x
        except Exception:
            pass
        
        if feasible:
            hi = mid
            best_S = mid
            best_config = best_local_config
        else:
            lo = mid
            
    inner = []
    for i in range(N):
        x = best_config[3*i]
        y = best_config[3*i+1]
        a = best_config[3*i+2]
        a = a % 360.0
        inner.append([x, y, a])
        
    return inner, (0.0, 0.0), best_S, 0.0

# EVOLVE-BLOCK-END
