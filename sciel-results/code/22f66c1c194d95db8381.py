import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11
EPS = 1e-9
START = time.time()
DEADLINE = 145.0  # safety margin before 150s kill

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
    """SAT: return True if convex polygons A and B overlap (including touching)."""
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
            if maxA < minB - 1e-7 or maxB < minA - 1e-7:
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
    Create a hexagonal lattice arrangement of 11 unit hexagons with a small gap.
    """
    inner = []
    inner.append([0.0, 0.0, 0.0])
    d = math.sqrt(3.0) * 1.01  # 1% gap
    for k in range(6):
        a = math.radians(30 + k * 60)
        inner.append([d * math.cos(a), d * math.sin(a), 0.0])
    d2 = 2 * d
    for k in range(4):
        a = math.radians(30 + k * 60)
        inner.append([d2 * math.cos(a), d2 * math.sin(a), 0.0])
    return inner, (0.0, 0.0), 0.0, 0.0

def _make_seed(rng):
    inner = [[rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0, 60)] for _ in range(N)]
    return inner, (0.0, 0.0), 10.0, 0.0

def _sat_overlap_penalty(A, B):
    """
    Compute a smooth penalty based on SAT separation.
    Returns 0 if separated, positive if overlapping.
    """
    min_pen = float('inf')
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            ax = poly[(i+1) % n][0] - poly[i][0]
            ay = poly[(i+1) % n][1] - poly[i][1]
            # Normalize normal
            length = math.sqrt(ax*ax + ay*ay)
            if length < 1e-12:
                continue
            nx, ny = -ay/length, ax/length
            minA = min(p[0]*nx + p[1]*ny for p in A)
            maxA = max(p[0]*nx + p[1]*ny for p in A)
            minB = min(p[0]*nx + p[1]*ny for p in B)
            maxB = max(p[0]*nx + p[1]*ny for p in B)
            # Separation on this axis
            sep = max(minB - maxA, minA - maxB)
            if sep < 0:
                # Overlapping on this axis, penalty is -sep (penetration depth)
                pen = -sep
                if pen < min_pen:
                    min_pen = pen
    if min_pen == float('inf'):
        return 0.0
    return max(0.0, min_pen)

def _de_objective(params):
    """
    Objective for Differential Evolution.
    params: [x1,y1,a1, ..., x11,y11,a11, ox, oy, rot]
    """
    pts = params[:N*3].reshape(N, 3)
    xs = pts[:, 0]
    ys = pts[:, 1]
    thetas = pts[:, 2]
    ox, oy = params[N*3], params[N*3+1]
    rot = params[N*3+2]
    
    # Compute outer hexagon side needed
    theta_rad = math.radians(rot)
    normals = []
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    
    max_proj = 0.0
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
    
    penalty = 0.0
    
    # Overlap penalty using SAT
    polys = []
    for k in range(N):
        a0 = math.radians(thetas[k])
        poly = []
        for m in range(6):
            a = a0 + math.radians(60 * m)
            poly.append((xs[k] + math.cos(a), ys[k] + math.sin(a)))
        polys.append(poly)
    
    for i in range(N):
        for j in range(i+1, N):
            pen = _sat_overlap_penalty(polys[i], polys[j])
            if pen > 0:
                penalty += pen * pen * 1000.0
    
    # Boundary penalty
    outer_s = side
    limit = outer_s * math.sqrt(3.0) / 2.0
    for k in range(N):
        a0 = math.radians(thetas[k])
        for m in range(6):
            a = a0 + math.radians(60 * m)
            px = xs[k] + math.cos(a)
            py = ys[k] + math.sin(a)
            for (nx, ny) in normals:
                proj = (px - ox) * nx + (py - oy) * ny
                if proj > limit:
                    penalty += (proj - limit) ** 2 * 1000.0
    
    return side + penalty

def construct():
    best = None
    best_score = float('inf')
    budget = START + DEADLINE
    rng = random.Random(42)
    np.random.seed(42)
    
    # Initial seed with gap
    seed_inner, seed_oc, seed_s, seed_rot = _hexagonal_seed()
    seed_s_tight = _min_outer_side(seed_inner, seed_oc, seed_rot)
    if _valid(seed_inner, seed_oc, seed_s_tight, seed_rot):
        best = (seed_inner, seed_oc, seed_s_tight, seed_rot)
        best_score = seed_s_tight
    
    # Define bounds for DE
    bounds = []
    for _ in range(N):
        bounds.append((-6, 6))
        bounds.append((-6, 6))
        bounds.append((0, 60))
    bounds.append((-3, 3))  # ox
    bounds.append((-3, 3))  # oy
    bounds.append((0, 60))  # rot
    
    # Run DE with multiple restarts
    for restart in range(5):
        if time.time() > budget - 30:
            break
        
        try:
            res = differential_evolution(
                _de_objective,
                bounds,
                seed=42 + restart,
                maxiter=30,
                popsize=20,
                tol=1e-5,
                atol=1e-5,
                polish=False,
                mutation=(0.5, 1.0),
                recombination=0.8
            )
            if res.fun < float('inf'):
                params = res.x
                pts = params[:N*3].reshape(N, 3)
                inner_de = [[float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])] for i in range(N)]
                oc_de = (float(params[N*3]), float(params[N*3+1]))
                rot_de = float(params[N*3+2])
                s_de = _min_outer_side(inner_de, oc_de, rot_de)
                
                # Validate
                if _valid(inner_de, oc_de, s_de, rot_de):
                    if s_de < best_score:
                        best_score = s_de
                        best = (inner_de, oc_de, s_de, rot_de)
        except Exception:
            pass
    
    # If we have a best, try to refine it with local search (Nelder-Mead)
    if best is not None and time.time() < budget - 10:
        inner_b, oc_b, s_b, rot_b = best
        init_params = np.array([p for (x, y, a) in inner_b for p in [x, y, a]] + [oc_b[0], oc_b[1], rot_b])
        
        # Try multiple Nelder-Mead restarts from slightly perturbed initial points
        for nm_restart in range(10):
            if time.time() > budget - 5:
                break
            
            if nm_restart == 0:
                start_params = init_params.copy()
            else:
                start_params = init_params + rng.gauss(0, 0.05, size=init_params.shape)
            
            try:
                res_nm = minimize(
                    _de_objective,
                    start_params,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                )
                if res_nm.fun < float('inf'):
                    params = res_nm.x
                    pts = params[:N*3].reshape(N, 3)
                    inner_nm = [[float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])] for i in range(N)]
                    oc_nm = (float(params[N*3]), float(params[N*3+1]))
                    rot_nm = float(params[N*3+2])
                    s_nm = _min_outer_side(inner_nm, oc_nm, rot_nm)
                    
                    if _valid(inner_nm, oc_nm, s_nm, rot_nm):
                        if s_nm < best_score:
                            best_score = s_nm
                            best = (inner_nm, oc_nm, s_nm, rot_nm)
            except Exception:
                pass
    
    if best is None:
        # Fallback to seed
        inner, oc, s, rot = _hexagonal_seed()
        s_tight = _min_outer_side(inner, oc, rot)
        return inner, (oc[0], oc[1]), s_tight, rot
    
    inner, oc, s, rot = best
    # Final tightening
    s_tight = _min_outer_side(inner, oc, rot)
    if s_tight < s and _valid(inner, oc, s_tight, rot):
        s = s_tight
    
    return inner, (oc[0], oc[1]), s, rot
