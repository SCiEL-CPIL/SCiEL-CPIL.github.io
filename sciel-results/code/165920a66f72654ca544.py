import math
import time
import random
import numpy as np

N = 11
START = time.time()
DEADLINE = 150.0

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
    # Normals to the sides of the outer hexagon
    # For a regular hexagon, the distance from center to side is S * sqrt(3) / 2
    # The normal direction for side k is at angle theta + 30 + k*60
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
    
    # S * sqrt(3)/2 >= max_proj  =>  S >= 2 * max_proj / sqrt(3)
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

def _hexagonal_seed():
    """
    Create a hexagonal lattice arrangement of 11 unit hexagons.
    For 11 hexagons: 1 center + 6 in first ring + 4 in second ring.
    """
    inner = []
    # Center
    inner.append([0.0, 0.0, 30.0])
    # First ring: 6 hexagons at distance sqrt(3) in 6 directions
    d = math.sqrt(3.0)
    for k in range(6):
        a = math.radians(30 + k * 60)
        inner.append([d * math.cos(a), d * math.sin(a), 30.0])
    # Second ring: 4 hexagons at distance 2*sqrt(3) in 4 of the 6 directions
    d2 = 2 * d
    for k in range(4):
        a = math.radians(30 + k * 60)
        inner.append([d2 * math.cos(a), d2 * math.sin(a), 30.0])
    
    return inner, (0.0, 0.0), 0.0, 30.0

def _hexagonal_seed2():
    """Alternative hexagonal arrangement with different 4th ring positions."""
    inner = []
    inner.append([0.0, 0.0, 30.0])
    d = math.sqrt(3.0)
    for k in range(6):
        a = math.radians(30 + k * 60)
        inner.append([d * math.cos(a), d * math.sin(a), 30.0])
    # Use different 4 positions in second ring
    d2 = 2 * d
    for k in [0, 1, 3, 4]:
        a = math.radians(30 + k * 60)
        inner.append([d2 * math.cos(a), d2 * math.sin(a), 30.0])
    return inner, (0.0, 0.0), 0.0, 30.0

def _make_seed(rng):
    inner = [[rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0, 60)] for _ in range(N)]
    return inner, (0.0, 0.0), 10.0, 0.0

def _try_refine(inner, outer_c, s, rot, rng, iters, budget):
    best = (inner, outer_c, s, rot)
    best_pen = 1.0 if not _valid(inner, outer_c, s, rot) else 0.0
    cur = (inner, outer_c, s, rot)
    cur_pen = best_pen
    
    # Tighten s first
    s_tight = _min_outer_side(cur[0], cur[1], cur[3])
    if s_tight < cur[2]:
        cur = (cur[0], cur[1], s_tight, cur[3])
        if _valid(cur[0], cur[1], cur[2], cur[3]):
            best = cur
            best_pen = 0.0
            cur_pen = 0.0
    
    it = 0
    while it < iters:
        if time.time() > budget:
            break
        it += 1
        # Perturb
        cand = [[x + rng.uniform(-0.3, 0.3), y + rng.uniform(-0.3, 0.3), a + rng.uniform(-15, 15)] for (x, y, a) in cur[0]]
        cand_oc = (cur[1][0] + rng.uniform(-0.2, 0.2), cur[1][1] + rng.uniform(-0.2, 0.2))
        cand_rot = cur[3] + rng.uniform(-10, 10)
        
        # Compute tight s for this candidate
        cand_s = _min_outer_side(cand, cand_oc, cand_rot)
        
        if _valid(cand, cand_oc, cand_s, cand_rot):
            cur = (cand, cand_oc, cand_s, cand_rot)
            cur_pen = 0.0
            if cand_s < best[2]:
                best = cur
                best_pen = 0.0
    
    return best

def construct():
    best = None
    best_score = float('inf')
    budget = START + DEADLINE - 5.0
    rng = random.Random(42)
    
    # Generate seeds
    seeds = []
    seeds.append(_hexagonal_seed())
    seeds.append(_hexagonal_seed2())
    for _ in range(8):
        seeds.append(_make_seed(rng))
    
    for seed in seeds:
        if time.time() > budget - 10:
            break
        inner, oc, s, rot = seed
        s_tight = _min_outer_side(inner, oc, rot)
        
        # Try local search refinement
        res = _try_refine(inner, oc, s_tight, rot, rng, iters=300, budget=budget - 5)
        inner_ls, oc_ls, s_ls, rot_ls = res
        if _valid(inner_ls, oc_ls, s_ls, rot_ls):
            if s_ls < best_score:
                best_score = s_ls
                best = (inner_ls, oc_ls, s_ls, rot_ls)
    
    # More restarts if needed
    restarts = 0
    while time.time() < budget - 2.0 and (best is None or best_score > 4.5):
        if time.time() > budget - 2.0:
            break
        restarts += 1
        inner, oc, s, rot = _make_seed(rng)
        s_tight = _min_outer_side(inner, oc, rot)
        res = _try_refine(inner, oc, s_tight, rot, rng, iters=200, budget=budget - 2)
        inner, oc, s, rot = res
        if _valid(inner, oc, s, rot):
            if s < best_score:
                best_score = s
                best = (inner, oc, s, rot)
        if restarts >= 30:
            break
    
    if best is None:
        # Fallback: hexagonal seed with tight bound
        inner, oc, s, rot = _hexagonal_seed()
        s_tight = _min_outer_side(inner, oc, rot)
        return inner, (oc[0], oc[1]), s_tight, rot
    
    inner, oc, s, rot = best
    # Final tightening
    s_tight = _min_outer_side(inner, oc, rot)
    if s_tight < s and _valid(inner, oc, s_tight, rot):
        s = s_tight
    
    return inner, (oc[0], oc[1]), s, rot
