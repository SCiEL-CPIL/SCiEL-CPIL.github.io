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
    for (x, y, a) in inner:
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
        # Extend 4 arms
        # Arms are at indices 0, 1, 2, 3, 4, 5 in Ring 1
        # Let's extend 0, 1, 3, 4 for 2-fold symmetry? Or 0, 2, 3, 5?
        # Let's try 0, 1, 3, 4
        for k in [0, 1, 3, 4]:
            a = math.radians(30 + k * 60)
            coords.append((2 * d * math.cos(a), 2 * d * math.sin(a)))
            
    elif topology == 'bays':
        # Fill 4 bays
        # Bay k is between Ring1[k] and Ring1[k+1]
        # Position: C_k + C_{k+1} - C_0
        for k in [0, 1, 3, 4]: # Symmetric selection
            a1 = math.radians(30 + k * 60)
            a2 = math.radians(30 + (k+1) * 60)
            c1 = (d * math.cos(a1), d * math.sin(a1))
            c2 = (d * math.cos(a2), d * math.sin(a2))
            coords.append((c1[0] + c2[0], c1[1] + c2[1]))
            
    elif topology == 'arms_opp':
        # Extend opposite arms 0, 3 and 1, 4?
        for k in [0, 1, 3, 4]:
            a = math.radians(30 + k * 60)
            coords.append((2 * d * math.cos(a), 2 * d * math.sin(a)))
            
    # Ensure we have 11
    return coords

def _rigid_cluster_side(coords, alpha_deg):
    # alpha_deg is the rotation of the OUTER hexagon relative to the cluster
    # Cluster is fixed at 0 rotation
    # We compute the max projection of cluster vertices onto outer normals
    
    # Get all vertices of the 11 hexagons
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
    # Minimize side length by varying alpha
    # Due to symmetry, we only need to search 0 to 30 degrees (or 60)
    # Let's search 0 to 60
    res = minimize_scalar(_rigid_cluster_side, bounds=(0, 60), args=(coords,), method='bounded')
    return res.fun, res.x

def construct():
    best_s = float('inf')
    best_inner = None
    best_oc = (0.0, 0.0)
    best_rot = 0.0
    
    topologies = ['arms', 'bays', 'arms_opp']
    
    for topo in topologies:
        coords = _get_cluster_coords(topo)
        if len(coords) != 11:
            continue
        s, alpha = _optimize_rigid_cluster(coords)
        if s < best_s:
            best_s = s
            # Reconstruct inner list
            best_inner = [[cx, cy, 0.0] for (cx, cy) in coords]
            best_rot = alpha # Outer rotation
            
    # Also try a general local search from the best rigid cluster
    if best_inner is not None:
        # Use scipy minimize on the full positions
        # Flatten inner
        init_params = []
        for (x, y, a) in best_inner:
            init_params.extend([x, y, a])
            
        def obj(params):
            pts = np.array(params).reshape(N, 3)
            inner = [[float(pts[i,0]), float(pts[i,1]), float(pts[i,2])] for i in range(N)]
            s = _min_outer_side(inner, (0.0, 0.0), best_rot)
            # Penalty for overlap
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = pts[i,0] - pts[j,0]
                    dy = pts[i,1] - pts[j,1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < math.sqrt(3.0) - 1e-6:
                        penalty += (math.sqrt(3.0) - dist)**2
            return s + 1000.0 * penalty
            
        try:
            res = minimize(obj, init_params, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < float('inf'):
                pts = res.x.reshape(N, 3)
                inner = [[float(pts[i,0]), float(pts[i,1]), float(pts[i,2])] for i in range(N)]
                s = _min_outer_side(inner, (0.0, 0.0), best_rot)
                if _valid(inner, (0.0, 0.0), s, best_rot):
                    if s < best_s:
                        best_s = s
                        best_inner = inner
        except Exception:
            pass

    if best_inner is None:
        # Fallback
        coords = _get_cluster_coords('bays')
        best_inner = [[cx, cy, 0.0] for (cx, cy) in coords]
        best_s = _min_outer_side(best_inner, (0.0, 0.0), 0.0)
        
    return best_inner, (0.0, 0.0), best_s, best_rot

