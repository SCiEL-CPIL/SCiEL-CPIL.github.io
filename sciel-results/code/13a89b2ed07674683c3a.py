import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
START = time.time()
DEADLINE = 145.0  # Leave 5s margin

def _hex_vertices(cx, cy, s, rot):
    """Vertices of a regular hexagon with center (cx,cy), side s, rotation rot (degrees)."""
    a0 = math.radians(rot)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return pts

def _inner_hex(cx, cy, ang_deg):
    """Unit regular hexagon (side 1) with center (cx,cy) and rotation ang_deg."""
    a0 = math.radians(ang_deg)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + math.cos(a), cy + math.sin(a)))
    return pts

def _min_outer_side(inner, outer_c, rot):
    """
    Compute the minimum side length S of a regular hexagon centered at outer_c
    with rotation rot that contains all inner hexagons.
    """
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
    """SAT: return True if convex polygons A and B overlap."""
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

def _honeycomb_seed():
    """
    Generate a honeycomb lattice seed for 11 hexagons.
    1 center + 6 ring 1 + 4 ring 2.
    """
    inner = []
    d = math.sqrt(3.0)
    # Center
    inner.append([0.0, 0.0, 0.0])
    # Ring 1
    for k in range(6):
        a = math.radians(30 + k * 60)
        inner.append([d * math.cos(a), d * math.sin(a), 0.0])
    # Ring 2 (only 4 to make 11)
    d2 = 2 * d
    for k in [0, 1, 3, 4]:
        a = math.radians(30 + k * 60)
        inner.append([d2 * math.cos(a), d2 * math.sin(a), 0.0])
    return inner

def _objective(params, rot, oc):
    """
    Objective function for optimization.
    params: flat array of length 3*N: [x0, y0, a0, x1, y1, a1, ...]
    """
    pts = params.reshape(N, 3)
    xs = pts[:, 0]
    ys = pts[:, 1]
    thetas = pts[:, 2]
    
    # Calculate outer side length
    theta_rad = math.radians(rot)
    normals = []
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    
    ox, oy = oc
    max_proj = 0.0
    # Pre-compute vertices for each hexagon for speed? 
    # For N=11, 66 vertices. 66*6 projections = 396 ops. Fast enough.
    for k in range(N):
        a0 = math.radians(thetas[k])
        for m in range(6):
            a = a0 + math.radians(60 * m)
            px = xs[k] + math.cos(a)
            py = ys[k] + math.sin(a)
            for (nx, ny) in normals:
                proj = (px - ox) * nx + (py - oy) * ny
                if proj > max_proj:
                    max_proj = proj
    
    side = 2.0 * max_proj / math.sqrt(3.0)
    
    # Overlap penalty
    # Approximate overlap by center distance for aligned hexagons is risky.
    # Use a softer penalty based on center distance.
    # Min distance between centers of non-overlapping aligned unit hexagons is sqrt(3).
    # If rotated, it can be smaller? No, the "width" is sqrt(3).
    # If we allow rotation, the min distance varies.
    # Let's use a penalty that grows quadratically if dist < sqrt(3).
    penalty = 0.0
    min_dist = math.sqrt(3.0)
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                penalty += (min_dist - dist)**2
    
    return side + 100.0 * penalty

def construct():
    best_inner = None
    best_oc = (0.0, 0.0)
    best_s = float('inf')
    best_rot = 0.0
    
    budget = START + DEADLINE
    
    # Try a few global rotations for the outer hexagon
    # The honeycomb lattice has 6-fold symmetry, so 0 to 30 degrees is sufficient
    rot_candidates = [0.0, 15.0, 30.0]
    
    # Seed: Honeycomb
    seed_inner = _honeycomb_seed()
    
    for rot in rot_candidates:
        if time.time() > budget - 10:
            break
            
        # Initial guess: Honeycomb seed
        init_params = np.array([p for (x, y, a) in seed_inner for p in [x, y, a]])
        
        # Optimize
        try:
            res = minimize(
                _objective,
                init_params,
                args=(rot, (0.0, 0.0)),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            
            if res.fun < float('inf'):
                pts = res.x.reshape(N, 3)
                cand_inner = [[float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])] for i in range(N)]
                cand_s = _min_outer_side(cand_inner, (0.0, 0.0), rot)
                
                # Check validity
                if _valid(cand_inner, (0.0, 0.0), cand_s, rot):
                    if cand_s < best_s:
                        best_s = cand_s
                        best_inner = cand_inner
                        best_rot = rot
                        best_oc = (0.0, 0.0)
                        
        except Exception:
            pass

    # If optimization failed or didn't find a valid solution, fall back to the raw seed
    if best_inner is None:
        best_inner = seed_inner
        best_s = _min_outer_side(best_inner, (0.0, 0.0), 0.0)
        best_rot = 0.0
        best_oc = (0.0, 0.0)

    return best_inner, best_oc, best_s, best_rot
