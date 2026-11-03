import math
import time
import random
import numpy as np
from scipy.optimize import minimize_scalar, minimize

N = 11
START = time.time()
DEADLINE = 150.0

def _hex_vertices(cx, cy, s, rot):
    a0 = math.radians(rot)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return pts

def _inner_hex(cx, cy, ang_deg):
    a0 = math.radians(ang_deg)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + math.cos(a), cy + math.sin(a)))
    return pts

def _min_outer_side(inner, outer_c, rot):
    theta_rad = math.radians(rot)
    normals = []
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    
    ox, oy = outer_c
    max_proj = 0.0
    for item in inner:
        x, y, a = item
        pts = _inner_hex(x, y, a)
        for (px, py) in pts:
            for (nx, ny) in normals:
                proj = (px - ox) * nx + (py - oy) * ny
                if proj > max_proj:
                    max_proj = proj
    
    return 2.0 * max_proj / math.sqrt(3.0)

def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def _point_in_poly(pt, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        c = _cross(poly[i], poly[(i+1) % n], pt)
        if abs(c) < 1e-9:
            continue
        s = 1 if c > 0 else -1
        if sign == 0:
            sign = s
        elif sign != s:
            return False
    return True

def _poly_overlap(A, B):
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            ax = poly[(i+1) % n][0] - poly[i][0]
            ay = poly[(i+1) % n][1] - poly[i][1]
            nx, ny = -ay, ax
            minA = min(p[0]*nx + p[1]*ny for p in A)
            maxA = max(p[0]*nx + p[1]*ny for p in A)
            minB = min(p[0]*nx + p[1]*ny for p in B)
            maxB = max(p[0]*nx + p[1]*ny for p in B)
            if maxA + 1e-9 < minB or maxB + 1e-9 < minA:
                return False
    return True

def _in_outer(inner_pts, outer_pts):
    return all(_point_in_poly(p, outer_pts) for p in inner_pts)

def _valid(inner, outer_c, s, rot):
    outer_pts = _hex_vertices(outer_c[0], outer_c[1], s, rot)
    polys = [_inner_hex(x, y, a) for (x, y, a) in inner]
    for i in range(len(polys)):
        if not _in_outer(polys[i], outer_pts):
            return False
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if _poly_overlap(polys[i], polys[j]):
                return False
    return True

def _get_cluster_coords(topology):
    d = math.sqrt(3.0)
    coords = [(0.0, 0.0)]
    
    # Ring 1
    for k in range(6):
        a = math.radians(30 + k * 60)
        coords.append((d * math.cos(a), d * math.sin(a)))
    
    if topology == 'arms':
        for k in [0, 1, 3, 4]:
            a = math.radians(30 + k * 60)
            coords.append((2 * d * math.cos(a), 2 * d * math.sin(a)))
            
    elif topology == 'bays':
        for k in [0, 1, 3, 4]:
            a1 = math.radians(30 + k * 60)
            a2 = math.radians(30 + (k+1) * 60)
            c1 = (d * math.cos(a1), d * math.sin(a1))
            c2 = (d * math.cos(a2), d * math.sin(a2))
            coords.append((c1[0] + c2[0], c1[1] + c2[1]))
            
    elif topology == 'arms_opp':
        for k in [0, 1, 3, 4]:
            a = math.radians(30 + k * 60)
            coords.append((2 * d * math.cos(a), 2 * d * math.sin(a)))
            
    return coords

def _rigid_cluster_side(coords, alpha_deg):
    all_pts = []
    for (cx, cy) in coords:
        pts = _inner_hex(cx, cy, 0.0)
        all_pts.extend(pts)
        
    theta_rad = math.radians(alpha_deg)
    max_proj = 0.0
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        nx = math.cos(alpha)
        ny = math.sin(alpha)
        for (px, py) in all_pts:
            proj = px * nx + py * ny
            if proj > max_proj:
                max_proj = proj
                
    return 2.0 * max_proj / math.sqrt(3.0)

def _optimize_rigid_cluster(coords):
    res = minimize_scalar(_rigid_cluster_side, bounds=(0, 60), args=(coords,), method='bounded')
    return float(res.fun), float(res.x)

def _sanitize_inner(inner):
    """Ensure inner is a list of lists of floats."""
    sanitized = []
    for item in inner:
        if isinstance(item, (list, tuple, np.ndarray)):
            sanitized.append([float(item[0]), float(item[1]), float(item[2])])
        else:
            # If it's a scalar, we can't fix it, but this shouldn't happen
            sanitized.append([0.0, 0.0, 0.0])
    return sanitized

def construct():
    best_s = float('inf')
    best_inner = None
    best_oc = (0.0, 0.0)
    best_rot = 0.0
    
    topologies = ['arms', 'bays', 'arms_opp']
    
    for topo in topologies:
        if time.time() > DEADLINE - 10:
            break
        coords = _get_cluster_coords(topo)
        if len(coords) != 11:
            continue
        s, alpha = _optimize_rigid_cluster(coords)
        if s < best_s:
            best_s = s
            best_inner = [[float(cx), float(cy), 0.0] for (cx, cy) in coords]
            best_rot = float(alpha)
            
    # Try local search if time permits
    if best_inner is not None and time.time() < DEADLINE - 30:
        init_params = []
        for (x, y, a) in best_inner:
            init_params.extend([x, y, a])
            
        def obj(params):
            pts = np.array(params).reshape(N, 3)
            inner = [[float(pts[i,0]), float(pts[i,1]), float(pts[i,2])] for i in range(N)]
            s = _min_outer_side(inner, (0.0, 0.0), best_rot)
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = pts[i,0] - pts[j,0]
                    dy = pts[i,1] - pts[j,1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < math.sqrt(3.0) - 1e-6:
                        penalty += (math.sqrt(3.0) - dist)**2
            return float(s) + 1000.0 * float(penalty)
            
        try:
            res = minimize(obj, init_params, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < float('inf'):
                pts = res.x.reshape(N, 3)
                inner = [[float(pts[i,0]), float(pts[i,1]), float(pts[i,2])] for i in range(N)]
                s = _min_outer_side(inner, (0.0, 0.0), best_rot)
                if _valid(inner, (0.0, 0.0), s, best_rot):
                    if s < best_s:
                        best_s = float(s)
                        best_inner = inner
        except Exception:
            pass

    if best_inner is None:
        coords = _get_cluster_coords('bays')
        best_inner = [[float(cx), float(cy), 0.0] for (cx, cy) in coords]
        best_s = _min_outer_side(best_inner, (0.0, 0.0), 0.0)
        
    best_inner = _sanitize_inner(best_inner)
    return best_inner, (0.0, 0.0), float(best_s), float(best_rot)
