# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_verts(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    th = a + np.radians(60.0 * np.arange(6))
    return np.stack([cx + s * np.cos(th), cy + s * np.sin(th)], axis=1)

def _width_dir(verts, n):
    p = verts @ n
    return p.max() - p.min()

def _sep_ok(h1, h2):
    """Check if two unit hexagons are separated. h=(x,y,ang_deg)."""
    x1, y1, a1 = h1
    x2, y2, a2 = h2
    v1 = _hex_verts(x1, y1, 1.0, a1)
    v2 = _hex_verts(x2, y2, 1.0, a2)
    d = np.array([x1 - x2, y1 - y2])
    # 12 normals
    for base_ang, verts in [(a1, v1), (a2, v2)]:
        for i in range(6):
            ang = np.radians(base_ang + 30.0 + 60.0 * i)
            n = np.array([np.cos(ang), np.sin(ang)])
            w1 = _width_dir(v1, n)
            w2 = _width_dir(v2, n)
            if abs(d @ n) > (w1 + w2) / 2.0 + 1e-6:
                return True
    return False

def _all_separated(inners):
    for i in range(11):
        for j in range(i + 1, 11):
            if not _sep_ok(inners[i], inners[j]):
                return False
    return True

def _in_outer(inners, outer_side):
    """Check all inners are inside outer hexagon at origin, angle 0, side=outer_side."""
    ov = _hex_verts(0.0, 0.0, outer_side, 0.0)
    for h in inners:
        iv = _hex_verts(h[0], h[1], 1.0, h[2])
        for i in range(6):
            p1 = ov[i]
            p2 = ov[(i + 1) % 6]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            nn = np.linalg.norm(n)
            if nn > 0:
                n = n / nn
            # Inward normal
            if np.dot(n, -p1) < 0:
                n = -n
            for p in iv:
                if np.dot(p - p1, n) < -1e-6:
                    return False
    return True

def _is_valid(inners, outer_side):
    if outer_side < np.sqrt(11) - 1e-9:
        return False
    if not np.all(np.isfinite(inners)):
        return False
    if not _all_separated(inners):
        return False
    if not _in_outer(inners, outer_side):
        return False
    return True

def _obj(params):
    inners = params[:33].reshape(11, 3)
    side = params[33]
    if side < np.sqrt(11):
        return 1e10
    # Penalty for violations
    pen = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            if not _sep_ok(inners[i], inners[j]):
                pen += 1.0
    if not _in_outer(inners, side):
        pen += 1.0
    return side + 50.0 * pen

def _init_sym():
    a = np.zeros((11, 3))
    a[0] = [0, 0, 0]
    for i in range(6):
        th = np.radians(60 * i)
        a[1 + i] = [2.5 * np.cos(th), 2.5 * np.sin(th), 0]
    for i in range(4):
        th = np.radians(45 + 90 * i)
        a[7 + i] = [4.5 * np.cos(th), 4.5 * np.sin(th), 0]
    return a

def _init_ring():
    a = np.zeros((11, 3))
    a[0] = [0, 0, 0]
    for i in range(6):
        th = np.radians(60 * i)
        a[1 + i] = [2.3 * np.cos(th), 2.3 * np.sin(th), 0]
    for i in range(4):
        th = np.radians(30 + 90 * i)
        a[7 + i] = [4.2 * np.cos(th), 4.2 * np.sin(th), 0]
    return a

def _opt(inners0, side0, tlimit):
    t0 = time.time()
    p = np.concatenate([inners0.ravel(), [side0]])
    bp, bo = p.copy(), _obj(p)
    for m in ['Nelder-Mead', 'Powell']:
        if time.time() - t0 > tlimit:
            break
        try:
            r = minimize(_obj, p, method=m, options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9})
            if r.fun < bo:
                bo, bp = r.fun, r.x
            p = r.x
        except:
            pass
    return bp, bo

def hexagon_packing_11():
    t0 = time.time()
    budget = 570.0
    best = None
    best_sc = -1e9
    
    inits = [(_init_sym(), 8.0), (_init_ring(), 7.5)]
    np.random.seed(123)
    for _ in range(2):
        a = np.random.randn(11, 3) * 2.5
        a[:, 2] = np.random.uniform(0, 60, 11)
        inits.append((a, 8.0))
    
    for a0, s0 in inits:
        rem = budget - (time.time() - t0)
        if rem < 5:
            break
        p, o = _opt(a0, s0, rem)
        inners = p[:33].reshape(11, 3)
        side = p[33]
        if _is_valid(inners, side):
            sc = 1.0 / side
            if sc > best_sc:
                best_sc = sc
                best = (inners.copy(), np.array([0.0, 0.0, 0.0]), side)
    
    if best is None:
        inners = _init_sym()
        best = (inners, np.array([0.0, 0.0, 0.0]), 8.0)
    return best[0], best[1], best[2]

# EVOLVE-BLOCK-END
