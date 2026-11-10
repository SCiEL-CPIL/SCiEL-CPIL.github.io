import numpy as np
import time
from scipy.optimize import minimize

def hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def point_in_hexagon_violation(pt, s, outer_angle_deg):
    a = s * np.sqrt(3) / 2
    base_angle = np.deg2rad(outer_angle_deg)
    normal_angles = base_angle + np.pi/6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    projections = pt[0]*nx + pt[1]*ny
    violations = np.maximum(0, projections - a)
    return np.sum(violations**2)

def dist_point_to_segment(p, a, b):
    p = np.array(p)
    a = np.array(a)
    b = np.array(b)
    ab = b - a
    ap = p - a
    denom = np.dot(ab, ab)
    if denom == 0:
        return np.linalg.norm(p - a)
    t = np.dot(ap, ab) / denom
    t = np.clip(t, 0, 1)
    proj = a + t * ab
    return np.linalg.norm(p - proj)

def min_dist_poly_poly(v1, v2):
    min_d = np.inf
    
    for i in range(6):
        p = v1[i]
        for j in range(6):
            a = v2[j]
            b = v2[(j+1)%6]
            d = dist_point_to_segment(p, a, b)
            if d < min_d:
                min_d = d
                
    for i in range(6):
        p = v2[i]
        for j in range(6):
            a = v1[j]
            b = v1[(j+1)%6]
            d = dist_point_to_segment(p, a, b)
            if d < min_d:
                min_d = d
                
    # Check intersection
    def segments_intersect(p1, p2, p3, p4):
        def ccw(a, b, c):
            return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)
        
    def point_in_poly(pt, poly):
        n = len(poly)
        inside = False
        p1 = poly[0]
        for i in range(1, n+1):
            p2 = poly[i % n]
            if (p1[1] > pt[1]) != (p2[1] > pt[1]):
                xint = (p2[0]-p1[0]) * (pt[1]-p1[1]) / (p2[1]-p1[1]) + p1[0]
                if pt[0] < xint:
                    inside = not inside
            p1 = p2
        return inside

    for i in range(6):
        a1 = v1[i]
        b1 = v1[(i+1)%6]
        for j in range(6):
            a2 = v2[j]
            b2 = v2[(j+1)%6]
            if segments_intersect(a1, b1, a2, b2):
                return 0.0
                
    for i in range(6):
        if point_in_poly(v1[i], v2):
            return 0.0
        if point_in_poly(v2[i], v1):
            return 0.0
            
    return min_d

def objective(params):
    n = 11
    s = params[33]
    outer_angle = params[34]
    
    if s < np.sqrt(11):
        s = np.sqrt(11)
        
    penalty = s
    
    # Centering penalty
    xs = params[0:33:3]
    ys = params[1:33:3]
    com_x = np.mean(xs)
    com_y = np.mean(ys)
    penalty += 0.1 * (com_x**2 + com_y**2)
    
    # Containment penalty
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        angle = params[3*i+2]
        verts = hex_vertices(cx, cy, angle)
        for v in verts:
            penalty += 100 * point_in_hexagon_violation(v, s, outer_angle)
            
    # Overlap penalty
    for i in range(n):
        cx1 = params[3*i]
        cy1 = params[3*i+1]
        a1 = params[3*i+2]
        v1 = hex_vertices(cx1, cy1, a1)
        for j in range(i+1, n):
            cx2 = params[3*j]
            cy2 = params[3*j+1]
            a2 = params[3*j+2]
            v2 = hex_vertices(cx2, cy2, a2)
            d = min_dist_poly_poly(v1, v2)
            if d < 1.0:
                penalty += 100 * (1.0 - d)
                
    return penalty

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 500
    
    init_params = []
    # 3-5-3 Grid
    # Row 0 (y=0): 5
    for k in range(-2, 3):
        x = k * np.sqrt(3)
        y = 0
        init_params.extend([x, y, 0])
        
    # Row 1 (y=1.5*sqrt(3)): 3
    for k in range(-1, 2):
        x = k * np.sqrt(3) + np.sqrt(3)/2
        y = np.sqrt(3) * 1.5
        init_params.extend([x, y, 0])
        
    # Row 2 (y=-1.5*sqrt(3)): 3
    for k in range(-1, 2):
        x = k * np.sqrt(3) + np.sqrt(3)/2
        y = -np.sqrt(3) * 1.5
        init_params.extend([x, y, 0])
        
    init_params.append(6.0)
    init_params.append(0.0)
    
    init_params = np.array(init_params)
    
    best_score = np.inf
    best_params = None
    
    for trial in range(5):
        if time.time() > deadline:
            break
            
        if trial == 0:
            p0 = init_params
        else:
            p0 = init_params + np.random.normal(0, 0.2, size=init_params.shape)
            
        res = minimize(objective, p0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            
    if best_params is None:
        inner = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0]
        ])
        outer = np.array([0, 0, 0])
        side = 8.0
        return inner, outer, side

    inner = np.zeros((11, 3))
    for i in range(11):
        inner[i, 0] = best_params[3*i]
        inner[i, 1] = best_params[3*i+1]
        inner[i, 2] = best_params[3*i+2]
        
    outer = np.array([0, 0, best_params[34]])
    side = best_params[33]
    
    return inner, outer, side
