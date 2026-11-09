import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_DEADLINE = 145.0
_C30 = math.cos(math.radians(30))
_S30 = math.sin(math.radians(30))

def _hex_inradius(s):
    return s * _C30

def _hex_verts(cx, cy, s, rot):
    vs = np.empty((6, 2))
    for k in range(6):
        a = math.radians(60.0 * k) + rot
        vs[k, 0] = cx + s * math.cos(a)
        vs[k, 1] = cy + s * math.sin(a)
    return vs

def _point_in_hex(px, py, cx, cy, s, rot):
    dx = px - cx
    dy = py - cy
    ca, sa = math.cos(-rot), math.sin(-rot)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    r_in = _hex_inradius(s)
    d1 = abs(_C30 * lx + 0.5 * ly)
    d2 = abs(ly)
    d3 = abs(-_C30 * lx + 0.5 * ly)
    return max(d1, d2, d3) <= r_in + 1e-9

def _min_dist_poly(A, B):
    """
    Compute minimum distance between two convex polygons.
    Returns 0.0 if overlapping.
    """
    nA = len(A)
    nB = len(B)
    
    def point_in_poly(pt, poly):
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % len(poly)]
            nx = -(y2 - y1)
            ny = (x2 - x1)
            d = nx * (pt[0] - x1) + ny * (pt[1] - y1)
            if d < -1e-9:
                return False
        return True
        
    # Check for intersection
    for v in A:
        if point_in_poly(v, B):
            return 0.0
    for v in B:
        if point_in_poly(v, A):
            return 0.0
            
    min_d = float('inf')
    for i in range(nA):
        ax1, ay1 = A[i]
        ax2, ay2 = A[(i+1) % nA]
        for j in range(nB):
            bx1, by1 = B[j]
            bx2, by2 = B[(j+1) % nB]
            
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
                
            d = min(pt_seg_dist(ax1, ay1, bx1, by1, bx2, by2),
                    pt_seg_dist(ax2, ay2, bx1, by1, bx2, by2),
                    pt_seg_dist(bx1, by1, ax1, ay1, ax2, ay2),
                    pt_seg_dist(bx2, by2, ax1, ay1, ax2, ay2))
            if d < min_d:
                min_d = d
    return min_d

def _penetration_depth(A, B):
    """
    Approximate penetration depth if polygons intersect.
    Returns a positive value representing how deep they overlap.
    """
    # Calculate center of A and B
    cx_a = np.mean(A[:, 0])
    cy_a = np.mean(A[:, 1])
    cx_b = np.mean(B[:, 0])
    cy_b = np.mean(B[:, 1])
    
    # Distance from center of A to nearest edge of B
    min_d_a = float('inf')
    for i in range(len(B)):
        x1, y1 = B[i]
        x2, y2 = B[(i+1) % len(B)]
        dx = x2 - x1
        dy = y2 - y1
        len_sq = dx*dx + dy*dy
        if len_sq == 0: continue
        t = ((cx_a - x1)*dx + (cy_a - y1)*dy) / len_sq
        t = max(0.0, min(1.0, t))
        qx = x1 + t*dx
        qy = y1 + t*dy
        d = math.sqrt((cx_a-qx)**2 + (cy_a-qy)**2)
        min_d_a = min(min_d_a, d)
        
    # Distance from center of B to nearest edge of A
    min_d_b = float('inf')
    for i in range(len(A)):
        x1, y1 = A[i]
        x2, y2 = A[(i+1) % len(A)]
        dx = x2 - x1
        dy = y2 - y1
        len_sq = dx*dx + dy*dy
        if len_sq == 0: continue
        t = ((cx_b - x1)*dx + (cy_b - y1)*dy) / len_sq
        t = max(0.0, min(1.0, t))
        qx = x1 + t*dx
        qy = y1 + t*dy
        d = math.sqrt((cx_b-qx)**2 + (cy_b-qy)**2)
        min_d_b = min(min_d_b, d)
        
    return min_d_a + min_d_b

def _is_valid_config(inner, center, side, angle):
    cx, cy = center
    s_out = side
    r_out = _hex_inradius(s_out)
    
    for i in range(len(inner)):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        verts = _hex_verts(xi, yi, 1.0, ai)
        for v in verts:
            if not _point_in_hex(v[0], v[1], cx, cy, s_out, angle):
                return False

    for i in range(len(inner)):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        vi = _hex_verts(xi, yi, 1.0, ai)
        for j in range(i + 1, len(inner)):
            xj, yj, aj_deg = inner[j]
            aj = math.radians(aj_deg)
            vj = _hex_verts(xj, yj, 1.0, aj)
            
            dc = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dc > 2.0:
                continue
            
            if _min_dist_poly(vi, vj) < 1e-7:
                return False
    return True

def _honeycomb_positions():
    positions = []
    positions.append((0.0, 0.0, 0.0))
    for k in range(6):
        a = math.radians(60.0 * k + 30.0)
        positions.append((2.0*math.cos(a), 2.0*math.sin(a), 0.0))
    for k in range(4):
        a = math.radians(60.0 * k + 30.0)
        positions.append((4.0*math.cos(a), 4.0*math.sin(a), 0.0))
    return positions[:N]

def construct():
    t_start = time.time()
    n = N

    def _remaining():
        return _DEADLINE - (time.time() - t_start)

    init_pos = _honeycomb_positions()
    
    max_r = 0.0
    for (xi, yi, ai) in init_pos:
        verts = _hex_verts(xi, yi, 1.0, ai)
        for v in verts:
            r = math.sqrt(v[0]**2 + v[1]**2)
            if r > max_r:
                max_r = r
    
    s_init = max_r / _C30
    s_init *= 1.01
    
    p0 = np.zeros(3*n + 1)
    for i in range(n):
        p0[3*i] = init_pos[i][0]
        p0[3*i+1] = init_pos[i][1]
        p0[3*i+2] = 0.0
    p0[3*n] = s_init
    
    def objective(p, beta_outer=10.0, beta_overlap=10.0):
        x = p[0:3*n:3]
        y = p[1:3*n:3]
        angles = p[2:3*n:3]
        S = p[3*n]
        
        if S < 1.0:
            return 1e6
            
        pen = 0.0
        r_out = _hex_inradius(S)
        
        # Boundary penalty using log-sum-exp for max
        # d1 = abs(_C30 * lx + 0.5 * ly) - r_out
        # d2 = abs(ly) - r_out
        # d3 = abs(-_C30 * lx + 0.5 * ly) - r_out
        # We want max(d1, d2, d3) <= 0
        
        sum_exp = 0.0
        for i in range(n):
            verts = _hex_verts(x[i], y[i], 1.0, angles[i])
            for v in verts:
                dx = v[0]
                dy = v[1]
                d1 = abs(_C30 * dx + 0.5 * dy) - r_out
                d2 = abs(dy) - r_out
                d3 = abs(-_C30 * dx + 0.5 * dy) - r_out
                # Only add if positive (violation)
                # Use soft max of positive parts
                # If d > 0, contribute to sum
                if d1 > 0: sum_exp += math.exp(beta_outer * d1)
                if d2 > 0: sum_exp += math.exp(beta_outer * d2)
                if d3 > 0: sum_exp += math.exp(beta_outer * d3)
                
        if sum_exp > 0:
            pen += (1.0 / beta_outer) * math.log(sum_exp)
        else:
            pen += 0.0
            
        # Overlap penalty
        for i in range(n):
            vi = _hex_verts(x[i], y[i], 1.0, angles[i])
            for j in range(i+1, n):
                dc = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dc > 2.5:
                    continue
                vj = _hex_verts(x[j], y[j], 1.0, angles[j])
                
                # Check intersection
                # Use a fast bounding box check first?
                # Just use min_dist
                d = _min_dist_poly(vi, vj)
                
                if d < 1e-7:
                    # Overlapping or touching
                    depth = _penetration_depth(vi, vj)
                    pen += 50.0 * depth * depth
                else:
                    # Separated, but add small penalty if too close to guide them apart?
                    # Not necessary if they are separated, but helps convergence
                    if d < 0.1:
                        pen += 5.0 * (0.1 - d)**2
                        
        return 0.5 * S * S + pen

    best_val = float('inf')
    best_p = p0.copy()
    
    rng = np.random.RandomState(42)
    
    # Optimization loop
    # Two stages: Soft then Hard
    stages = [
        {'beta_o': 5.0, 'beta_ov': 5.0, 'iters': 100, 'noise': 0.2},
        {'beta_o': 20.0, 'beta_ov': 20.0, 'iters': 200, 'noise': 0.0},
    ]
    
    for trial in range(6):
        if _remaining() < 20.0:
            break
            
        p_init = p0.copy()
        if trial > 0:
            noise = 0.1
            p_init[0:3*n] += rng.randn(3*n) * noise
            p_init[3*n] = s_init * (1.0 + rng.rand()*0.05)
            
        # Stage 1: Soft
        if _remaining() < 10.0: continue
        try:
            res1 = minimize(lambda p: objective(p, stages[0]['beta_o'], stages[0]['beta_ov']), 
                            p_init, method='Nelder-Mead',
                            options={'maxiter': stages[0]['iters'], 'xatol': 1e-3, 'fatol': 1e-3})
            p_init = res1.x
        except Exception:
            pass
            
        # Stage 2: Hard
        if _remaining() < 5.0: continue
        try:
            res2 = minimize(lambda p: objective(p, stages[1]['beta_o'], stages[1]['beta_ov']), 
                            p_init, method='Nelder-Mead',
                            options={'maxiter': stages[1]['iters'], 'xatol': 1e-4, 'fatol': 1e-4})
            
            # Evaluate with high beta to check validity
            val = objective(res2.x, 50.0, 50.0)
            if val < best_val:
                best_val = val
                best_p = res2.x.copy()
        except Exception:
            pass

    x = best_p[0:3*n:3]
    y = best_p[1:3*n:3]
    angles = best_p[2:3*n:3]
    S = best_p[3*n]
    
    inner = [[float(x[i]), float(y[i]), float(math.degrees(angles[i]))] for i in range(n)]
    center = (0.0, 0.0)
    angle_out = 0.0
    
    if _is_valid_config(inner, center, S, angle_out):
        return inner, center, float(S), angle_out
    else:
        inner_init = [[float(p[0]), float(p[1]), float(math.degrees(p[2]))] for p in init_pos]
        return inner_init, center, float(s_init), angle_out
