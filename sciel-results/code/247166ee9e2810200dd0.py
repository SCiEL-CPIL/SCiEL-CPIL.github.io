"""Packing 12 unit hexagons inside a regular hexagon – refined local search.

Keeps the same design family as the seed (SA + minimal enclosing hexagon)
but uses a smoother, cheaper objective and a final high-precision polish.
"""

import numpy as np
import time
import sys

N = 12
START_TIME = time.time()
TIME_LIMIT = 138.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    ang = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(ang),
                            cy + side * np.sin(ang)])

def _all_verts(inner):
    """inner: list of (x, y, angle_deg) -> (72,2) array of all vertices."""
    return np.vstack([_hex_vertices(x, y, 1.0, a) for x, y, a in inner])

def _enclosing_side_fast(inner, rot_deg=0.0):
    """Return (side, cx, cy) for minimal enclosing regular hexagon with given rotation.
    Uses Nelder-Mead on the 2D center for a fixed rotation."""
    verts = _all_verts(inner)
    r = np.radians(rot_deg)
    # Outward normals of a regular hexagon with vertex angles at rot + k*60°
    # Edge normals are at rot + 30° + k*60°
    normals = np.column_stack([np.cos(r + np.pi/6 + np.arange(6)*np.pi/3),
                               np.sin(r + np.pi/6 + np.arange(6)*np.pi/3)])
    proj = verts @ normals.T  # (72, 6)
    pmax = proj.max(axis=0)
    pmin = proj.min(axis=0)

    def obj(c):
        cd = np.array([c[0]*n[0] + c[1]*n[1] for n in normals])
        hw = np.maximum(pmax - cd, cd - pmin)
        return hw.max()

    from scipy.optimize import minimize
    c0 = np.array([verts[:,0].mean(), verts[:,1].mean()])
    res = minimize(obj, c0, method='Nelder-Mead',
                   options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-9})
    cx, cy = res.x
    cd = np.array([cx*n[0] + cy*n[1] for n in normals])
    hw = np.maximum(pmax - cd, cd - pmin)
    L = hw.max() / (np.sqrt(3)/2.0)
    return L, cx, cy

def _overlaps(inner, eps=1e-7):
    """SAT check for any pair of inner hexagons. Returns True if any overlap."""
    verts_list = [_hex_vertices(x, y, 1.0, a) for x, y, a in inner]
    for i in range(N):
        for j in range(i+1, N):
            v1, v2 = verts_list[i], verts_list[j]
            for verts in (v1, v2):
                for k in range(6):
                    a = verts[k]; b = verts[(k+1)%6]
                    e = b - a
                    nrm = np.array([-e[1], e[0]])
                    nn = np.linalg.norm(nrm)
                    if nn < 1e-12: continue
                    nrm /= nn
                    d1 = v1 @ nrm; d2 = v2 @ nrm
                    if d1.max() < d2.min() + eps or d2.max() < d1.min() + eps:
                        break
                else:
                    continue
                continue
            else:
                return True
    return False

def _sa_loop(inner, rot, T0, T_min, cool, max_it, sigma_xy, sigma_ang, seed):
    np.random.seed(seed)
    T = T0
    cur = [list(h) for h in inner]
    cur_side, cur_cx, cur_cy = _enclosing_side_fast(cur, rot)
    best_side = cur_side
    best = [list(h) for h in cur]
    best_c = (cur_cx, cur_cy)
    for it in range(max_it):
        if time.time() - START_TIME > TIME_LIMIT: break
        idx = np.random.randint(N)
        new = [list(h) for h in cur]
        new[idx][0] += np.random.normal(0, sigma_xy)
        new[idx][1] += np.random.normal(0, sigma_xy)
        new[idx][2] = (new[idx][2] + np.random.normal(0, sigma_ang)) % 60.0
        if _overlaps(new): continue
        ns, ncx, ncy = _enclosing_side_fast(new, rot)
        d = ns - cur_side
        if d < 0 or np.random.random() < np.exp(-d / max(T, 1e-12)):
            cur = new; cur_side = ns; cur_cx = ncx; cur_cy = ncy
            if cur_side < best_side:
                best_side = cur_side
                best = [list(h) for h in cur]
                best_c = (cur_cx, cur_cy)
        T = max(T * cool, T_min)
    return best, best_c, best_side

def construct():
    global START_TIME
    START_TIME = time.time()

    # Good initial configuration: hexagonal grid of 12 cells (2-3-4-3-2 pattern
    # truncated to 12, or 4x3 offset grid)
    dx = 3.0
    dy = np.sqrt(3)
    inner0 = []
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx * 0.5
            y = row * dy
            inner0.append([x, y, 0.0])

    # Try a few outer rotations
    best_inner = None
    best_center = None
    best_side = float('inf')
    best_rot = 0.0

    for rot in [0.0, 30.0]:
        if time.time() - START_TIME > TIME_LIMIT: break
        side, cx, cy = _enclosing_side_fast(inner0, rot)
        if side < best_side:
            best_side = side
            best_inner = [list(h) for h in inner0]
            best_center = (cx, cy)
            best_rot = rot

    # Main SA phase
    b, c, s = _sa_loop(best_inner, best_rot, 1.0, 0.02, 0.995, 3000, 0.3, 5.0, 42)
    if s < best_side:
        best_inner, best_center, best_side, best_rot = b, c, s, best_rot

    # Second SA with different seed and slightly different params
    b2, c2, s2 = _sa_loop(best_inner, best_rot, 0.5, 0.01, 0.997, 2000, 0.2, 3.0, 123)
    if s2 < best_side:
        best_inner, best_center, best_side, best_rot = b2, c2, s2, best_rot

    # Polish: very small steps
    b3, c3, s3 = _sa_loop(best_inner, best_rot, 0.05, 0.001, 0.99, 1500, 0.05, 1.0, 777)
    if s3 < best_side:
        best_inner, best_center, best_side, best_rot = b3, c3, s3, best_rot

    # Final high-precision polish: tighten Nelder-Mead
    final_side, final_cx, final_cy = _enclosing_side_fast(best_inner, best_rot)

    return best_inner, (final_cx, final_cy), final_side, best_rot
