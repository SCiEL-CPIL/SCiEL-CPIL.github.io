"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def hex_vertices(cx, cy, side, angle_deg):
    angle = np.deg2rad(angle_deg)
    angles = np.arange(6) * np.pi / 3 + angle
    x = side * np.cos(angles)
    y = side * np.sin(angles)
    verts = np.column_stack((x, y))
    verts[:, 0] += cx
    verts[:, 1] += cy
    return verts

def point_in_polygon(pt, poly):
    # poly is Nx2, CCW
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1)%n]
        ex = x2 - x1
        ey = y2 - y1
        px = pt[0] - x1
        py = pt[1] - y1
        cross = ex * py - ey * px
        if cross < 0:
            return False
    return True

def dist_point_to_segment(pt, p1, p2):
    v = p2 - p1
    w = pt - p1
    c1 = np.dot(w, v)
    if c1 <= 0:
        return np.linalg.norm(pt - p1)
    c2 = np.dot(v, v)
    if c2 <= c1:
        return np.linalg.norm(pt - p2)
    b = c1 / c2
    pb = p1 + b * v
    return np.linalg.norm(pt - pb)

def dist_point_to_polygon(pt, poly):
    min_d = np.inf
    n = len(poly)
    for i in range(n):
        d = dist_point_to_segment(pt, poly[i], poly[(i+1)%n])
        if d < min_d:
            min_d = d
    return min_d

def penalty_and_objective(params):
    xs = params[0:N]
    ys = params[N:2*N]
    as_ = params[2*N:3*N]
    L = params[3*N]
    
    if L < 2.0: L = 2.0
    
    inner_verts = []
    for i in range(N):
        v = hex_vertices(xs[i], ys[i], 1.0, as_[i])
        inner_verts.append(v)
        
    outer_verts = hex_vertices(0.0, 0.0, L, 0.0)
    
    penalty = 0.0
    
    for i in range(N):
        for v in inner_verts[i]:
            if not point_in_polygon(v, outer_verts):
                d = dist_point_to_polygon(v, outer_verts)
                penalty += d * d
                
    for i in range(N):
        for j in range(i+1, N):
            for v in inner_verts[i]:
                if point_in_polygon(v, inner_verts[j]):
                    d = dist_point_to_polygon(v, inner_verts[j])
                    penalty += d * d
            for v in inner_verts[j]:
                if point_in_polygon(v, inner_verts[i]):
                    d = dist_point_to_polygon(v, inner_verts[i])
                    penalty += d * d
                    
    lam = 100.0
    return L + lam * penalty

def construct():
    start_time = time.time()
    deadline = 120.0
    
    coords = [
        (0,0),
        (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
        (2,-1), (2,0), (-2,1), (-2,0)
    ]
    
    xs = []
    ys = []
    as_ = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        xs.append(x)
        ys.append(y)
        as_.append(0.0)
        
    max_r = max(np.sqrt(x*x + y*y) for x, y in zip(xs, ys))
    L_init = max_r + 1.5
    
    x0 = np.array(xs + ys + as_ + [L_init])
    
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0))
        bounds.append((-5.0, 5.0))
        bounds.append((0.0, 360.0))
    bounds.append((2.0, 10.0))
    
    best_score = np.inf
    best_params = x0
    
    try:
        res = minimize(penalty_and_objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8})
        
        if time.time() > deadline - 10:
            pass
        
        if res.fun < best_score:
            best_params = res.x
            best_score = res.fun
            
        L_opt = best_params[3*N]
        
        p = penalty_and_objective(best_params)
        penalty_val = (p - L_opt) / 100.0
        
        if penalty_val < 1e-4:
            inner = []
            for i in range(N):
                inner.append([best_params[i], best_params[N+i], best_params[2*N+i]])
            return inner, (0.0, 0.0), L_opt, 0.0
    except Exception as e:
        pass
        
    p0 = penalty_and_objective(x0)
    L0 = x0[3*N]
    pen0 = (p0 - L0) / 100.0
    if pen0 < 1e-4:
        inner = []
        for i in range(N):
            inner.append([x0[i], x0[N+i], x0[2*N+i]])
        return inner, (0.0, 0.0), L0, 0.0
        
    inner = []
    for i in range(N):
        inner.append([4.0 * i, 0.0, 0.0])
    return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END
