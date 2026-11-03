"""Optimized hex packing: verified geometry + efficient LP + topology search."""
import math
import time
import numpy as np
import itertools
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0

def _hex_verts(cx, cy, ang_deg):
    """6 vertices of unit hex centered at (cx,cy), rotated ang_deg."""
    ar = math.radians(ang_deg)
    return [(cx + math.cos(ar + math.pi*k/3), cy + math.sin(ar + math.pi*k/3)) for k in range(6)]

def _all_verts(inner):
    """All 6N vertices as (6N,2) array."""
    out = np.zeros((len(inner)*6, 2))
    for i, (x, y, a) in enumerate(inner):
        for k in range(6):
            ar = math.radians(a) + math.pi*k/3
            out[i*6+k] = [x + math.cos(ar), y + math.sin(ar)]
    return out

def _enc_hex_lp(verts, outer_ang):
    """Min r s.t. for all 6 directions k, max_i n_k.(v_i - c) <= r.
    LP vars: [cx, cy, r]. Returns (cx, cy, r)."""
    n = verts.shape[0]
    if n == 0:
        return 0.0, 0.0, 0.0
    oa = math.radians(outer_ang)
    # Outer hex normals (perpendicular to edges, pointing outward)
    # For a regular hex with "flat top" at angle 0, normals are at 0,60,...,300
    # With rotation outer_ang, normals at outer_ang + 60*k
    normals = np.array([[math.cos(oa + math.pi*k/3), math.sin(oa + math.pi*k/3)] for k in range(6)])
    # Constraint: n_k.(v_i - c) <= r for all i,k
    # => -n_kx*cx - n_ky*cy - r <= -n_k.v_i
    A_rows = []
    b_rows = []
    for k in range(6):
        nk = normals[k]
        proj = verts @ nk  # (n,)
        A_k = np.column_stack([-np.full(n, nk[0]), -np.full(n, nk[1]), -np.ones(n)])
        b_k = -proj
        A_rows.append(A_k)
        b_rows.append(b_k)
    A_ub = np.vstack(A_rows)
    b_ub = np.concatenate(b_rows)
    A_ub = np.vstack([A_ub, [[0., 0., -1.]]])  # r >= 0
    b_ub = np.append(b_ub, 0.)
    c = [0., 0., 1.]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    if res.success:
        return float(res.x[0]), float(res.x[1]), float(res.x[2])
    # Fallback
    cx, cy = verts[:,0].mean(), verts[:,1].mean()
    r = 0.
    for k in range(6):
        proj = (verts - np.array([cx,cy])) @ normals[k]
        r = max(r, proj.max())
    return cx, cy, r

def _side_from_r(r):
    return r / (math.sqrt(3)/2)

def _min_side(verts, outer_ang):
    cx, cy, r = _enc_hex_lp(verts, outer_ang)
    return _side_from_r(r), cx, cy

def _enclose_best(inner):
    """Find best outer hex: sweep angle, refine."""
    verts = _all_verts(inner)
    best_s = float('inf')
    best_cx = best_cy = best_a = 0.
    
    # Coarse: 0 to 60 step 3
    for a in range(0, 60, 3):
        s, cx, cy = _min_side(verts, float(a))
        if s < best_s:
            best_s, best_cx, best_cy, best_a = s, cx, cy, float(a)
    
    # Fine: around best
    for a in np.arange(best_a-3, best_a+3, 0.25):
        a = float(a % 60)
        s, cx, cy = _min_side(verts, a)
        if s < best_s:
            best_s, best_cx, best_cy, best_a = s, cx, cy, a
    
    # NM polish on angle
    if time.time() < _DEADLINE - 25:
        def obj(a):
            return _min_side(verts, float(a % 60))[0]
        try:
            res = minimize(obj, [best_a], method='Nelder-Mead',
                         options={'maxiter':200,'xatol':1e-11,'fatol':1e-13,'adaptive':True})
            if res.fun < best_s:
                best_a = float(res.x[0] % 60)
                best_s = res.fun
                _, best_cx, best_cy = _min_side(verts, best_a)
        except:
            pass
    
    # High-res grid
    if time.time() < _DEADLINE - 10:
        for a in np.linspace(best_a-0.3, best_a+0.3, 61):
            a = float(a % 60)
            s, cx, cy = _min_side(verts, a)
            if s < best_s:
                best_s, best_cx, best_cy, best_a = s, cx, cy, a
    
    return best_cx, best_cy, best_s, best_a

def construct():
    if time.time() > _DEADLINE - 5:
        inner = [[4*k, 0.0, 30.0] for k in range(-(N//2), N-N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    sqrt3 = math.sqrt(3)
    # Unit hexagons rotated 30° (pointy-top). 
    # Touching distance = 2*apothem = 2*(sqrt(3)/2) = sqrt(3)
    d = sqrt3
    
    # Hex lattice: a1 = (d, 0), a2 = (d/2, d*sqrt(3)/2) = (d/2, 3/2)
    a1 = np.array([d, 0.0])
    a2 = np.array([d/2, 1.5])  # since d*sqrt(3)/2 = sqrt(3)*sqrt(3)/2 = 3/2
    
    # Generate lattice points
    pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i*a1 + j*a2
            r2 = p[0]**2 + p[1]**2
            if r2 <= (2.2*d)**2:
                pts.append((float(p[0]), float(p[1]), r2))
    
    pts.sort(key=lambda x: x[2])
    
    # Central 7
    central = [(x, y) for x, y, _ in pts[:7]]
    
    # Ring 2 candidates (positions 7-18)
    ring2 = [(x, y) for x, y, _ in pts[7:19]]
    
    inner_ang = 30.0
    best_s = float('inf')
    best_inner = None
    best_cx = best_cy = best_a = 0.
    
    # Try all C(12,4) = 495 combos
    for combo in itertools.combinations(range(12), 4):
        if time.time() > _DEADLINE - 45:
            break
        sel = [ring2[c] for c in combo]
        all_pts = central + sel
        inner = [[x, y, inner_ang] for x, y in all_pts]
        
        cx, cy, s, a = _enclose_best(inner)
        if s < best_s:
            best_s = s
            best_inner = inner
            best_cx, best_cy, best_a = cx, cy, a
    
    if best_inner is None:
        all_pts = [(x, y) for x, y, _ in pts[:11]]
        best_inner = [[x, y, inner_ang] for x, y in all_pts]
        best_cx, best_cy, best_s, best_a = _enclose_best(best_inner)
    
    return best_inner, (best_cx, best_cy), best_s, best_a
