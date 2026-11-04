import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
START = time.time()
DEADLINE = 148.0

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

def _valid(inner, outer_c, s, rot):
    # For rigid clusters, we can skip the overlap check if we trust the lattice,
    # but we must ensure containment.
    # However, _min_outer_side guarantees containment for the returned s.
    # So if we return s = _min_outer_side(...), it is valid.
    # We just need to check if the inner hexagons overlap each other.
    # In our rigid lattice construction, they do not.
    # But to be safe, we can run the original _valid.
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
            if maxA + 1e-6 < minB or maxB + 1e-6 < minA:
                return False
    return True

def _in_outer(inner_pts, outer_pts):
    return all(_point_in_poly(p, outer_pts) for p in inner_pts)

def _generate_cluster(type_idx):
    # Returns list of (x, y) relative to cluster center
    # Lattice: x = sqrt(3)*(q + r/2), y = 1.5*r
    sq3 = math.sqrt(3.0)
    pts = []
    
    # Center
    pts.append((0.0, 0.0))
    
    # Ring 1 (6 points)
    ring1_axial = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for q, r in ring1_axial:
        x = sq3 * (q + r/2.0)
        y = 1.5 * r
        pts.append((x, y))
        
    # Ring 2 (4 points)
    # We select 4 from the 12 available to form a compact shape.
    # Option 0: Symmetric corners
    # (1,1), (-1,2), (-2,1), (0,-2) -> Indices 1, 2, 3, 6 in a sorted list?
    # Let's just hardcode 3 different sets of 4 axial coords.
    
    if type_idx == 0:
        ring2_axial = [(1,1), (-1,2), (-2,1), (0,-2)]
    elif type_idx == 1:
        ring2_axial = [(1,1), (-1,2), (0,2), (-2,1)] # Slightly different
    elif type_idx == 2:
        ring2_axial = [(2,0), (1,1), (-1,2), (-2,1)] # Includes a side point
    else:
        ring2_axial = [(1,1), (-1,2), (-2,1), (-1,0)] # Invalid, (-1,0) is ring 1. 
        # Let's use (1,-2) instead of (0,-2) for variety
        ring2_axial = [(1,1), (-1,2), (-2,1), (1,-2)]

    for q, r in ring2_axial:
        x = sq3 * (q + r/2.0)
        y = 1.5 * r
        pts.append((x, y))
        
    return pts

def _objective_rigid(params, cluster_pts):
    # params: [x, y, rot_deg]
    # Outer hexagon is fixed at (0,0) with rotation 0.
    x, y, rot_deg = params
    rot_rad = math.radians(rot_deg)
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    
    inner = []
    for px, py in cluster_pts:
        # Rotate
        nx = px * cos_r - py * sin_r
        ny = px * sin_r + py * cos_r
        # Translate
        nx += x
        ny += y
        inner.append([nx, ny, rot_deg])
        
    s = _min_outer_side(inner, (0.0, 0.0), 0.0)
    return s

def construct():
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = float('inf')
    best_rot = 0.0
    
    budget = START + DEADLINE - 2.0
    
    # Try multiple cluster types
    for c_type in range(3):
        if time.time() > budget:
            break
            
        cluster_pts = _generate_cluster(c_type)
        
        # Initial guess: center at 0,0, rot 0
        x0 = [0.0, 0.0, 0.0]
        
        # Run Nelder-Mead
        try:
            res = minimize(
                _objective_rigid,
                x0,
                args=(cluster_pts,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res.fun < float('inf'):
                x_opt, y_opt, rot_opt = res.x
                rot_opt_deg = rot_opt % 60.0 # Normalize? No, keep as is for consistency
                
                # Construct inner list
                rot_rad = math.radians(rot_opt)
                cos_r = math.cos(rot_rad)
                sin_r = math.sin(rot_rad)
                
                inner = []
                for px, py in cluster_pts:
                    nx = px * cos_r - py * sin_r + x_opt
                    ny = px * sin_r + py * cos_r + y_opt
                    inner.append([nx, ny, rot_opt])
                    
                s_opt = _min_outer_side(inner, (0.0, 0.0), 0.0)
                
                # Verify validity (just in case)
                if _valid(inner, (0.0, 0.0), s_opt, 0.0):
                    if s_opt < best_side:
                        best_side = s_opt
                        best_inner = inner
                        best_center = (0.0, 0.0)
                        best_rot = 0.0
                        
        except Exception as e:
            pass
            
    # Fallback to hexagonal seed if optimization failed
    if best_inner is None:
        inner, oc, s, rot = _hexagonal_seed()
        s_tight = _min_outer_side(inner, oc, rot)
        if _valid(inner, oc, s_tight, rot):
            return inner, (oc[0], oc[1]), s_tight, rot
        else:
            return inner, (oc[0], oc[1]), s, rot
            
    return best_inner, best_center, best_side, best_rot

def _hexagonal_seed():
    inner = []
    inner.append([0.0, 0.0, 0.0])
    d = math.sqrt(3.0)
    for k in range(6):
        a = math.radians(30 + k * 60)
        inner.append([d * math.cos(a), d * math.sin(a), 0.0])
    d2 = 2 * d
    for k in range(4):
        a = math.radians(30 + k * 60)
        inner.append([d2 * math.cos(a), d2 * math.sin(a), 0.0])
    return inner, (0.0, 0.0), 0.0, 0.0
