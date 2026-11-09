import numpy as np
import time
from scipy.optimize import minimize

N = 11
DEADLINE = 150.0
START = time.time()

def _elapsed():
    return time.time() - START

# Outer (fixed) hexagon: 6 outward unit normals, inradius = side * INR
OUTER_N = np.array([[np.cos(np.radians(60*k)), np.sin(np.radians(60*k))] for k in range(6)])
INR = np.sqrt(3.0) / 2.0
# Unit hexagon base vertices (pointy-top), radius 1
BASE_V = np.array([[np.cos(np.radians(60*k)), np.sin(np.radians(60*k))] for k in range(6)])

def _hex_verts(x, y, ang):
    a = np.radians(ang)
    ca, sa = np.cos(a), np.sin(a)
    R = np.array([[ca, -sa], [sa, ca]])
    return (R @ BASE_V.T).T + np.array([x, y])

def _edge_normals(V):
    n = np.empty((6, 2))
    for k in range(6):
        e = V[(k + 1) % 6] - V[k]
        nn = np.array([-e[1], e[0]])
        nn /= (np.linalg.norm(nn) + 1e-12)
        n[k] = nn
    return n

def _penetration(A, B):
    """Max projection overlap over edge normals; >0 iff overlapping."""
    axes = np.vstack([_edge_normals(A), _edge_normals(B)])
    best = 0.0
    for u in axes:
        a_min, a_max = (A @ u).min(), (A @ u).max()
        b_min, b_max = (B @ u).min(), (B @ u).max()
        ov = min(a_max, b_max) - max(a_min, b_min)
        if ov > best:
            best = ov
    return best

def _verts(params):
    return [_hex_verts(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]

def _outer_side(vl):
    proj = np.vstack(vl) @ OUTER_N.T
    return proj.max() / INR

def _max_pen(vl):
    m = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            p = _penetration(vl[i], vl[j])
            if p > m:
                m = p
    return m

def _sum_pen_sq(vl):
    s = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            p = _penetration(vl[i], vl[j])
            if p > 0:
                s += p * p
    return s

def _obj_smooth(params, lam, beta=30.0):
    vl = _verts(params)
    proj = np.vstack(vl) @ OUTER_N.T
    pmax = proj.max()
    outer = (pmax + np.log(np.exp((proj - pmax) * beta).sum()) / beta) / INR
    return outer + lam * _sum_pen_sq(vl)

def _obj_hard(params, lam):
    vl = _verts(params)
    return _outer_side(vl) + lam * _sum_pen_sq(vl)

def _init_seeds():
    seeds = []
    s = np.zeros(3 * N)
    s[0:3] = [0.0, 0.0, 0.0]
    for k in range(6):
        ang = 60 * k
        s[3*(k+1):3*(k+2)] = [2*np.cos(np.radians(ang)), 2*np.sin(np.radians(ang)), 0.0]
    for k, ang in enumerate([30, 90, 210, 270]):
        idx = 7 + k
        s[3*idx:3*idx+3] = [4*np.cos(np.radians(ang)), 4*np.sin(np.radians(ang)), 0.0]
    seeds.append(s)
    s2 = s.copy()
    for i in range(1, N):
        s2[3*i + 2] = 12.0
    seeds.append(s2)
    s3 = s.copy()
    for i in range(1, N):
        s3[3*i + 2] = -12.0
    seeds.append(s3)
    return seeds

def construct():
    best = None
    best_side = np.inf
    for seed in _init_seeds():
        if _elapsed() > DEADLINE - 30:
            break
        cur = seed.copy()
        # Stage 1: smooth objective, ramp penalty weight
        lam = 1.0
        while _elapsed() < DEADLINE - 25:
            res = minimize(_obj_smooth, cur, args=(lam,), method='Nelder-Mead',
                           options={'maxiter': 1200, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            cur = res.x
            vl = _verts(cur)
            if _max_pen(vl) <= 1e-6:
                side = _outer_side(vl)
                if side < best_side:
                    best_side = side
                    best = cur.copy()
            lam = min(lam * 2, 100.0)
            if lam >= 100.0:
                break
        # Stage 2: hard objective, high weight, a few passes
        for _ in range(3):
            if _elapsed() > DEADLINE - 20:
                break
            res = minimize(_obj_hard, cur, args=(1000.0,), method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            cur = res.x
            vl = _verts(cur)
            if _max_pen(vl) <= 1e-6:
                side = _outer_side(vl)
                if side < best_side:
                    best_side = side
                    best = cur.copy()
    if best is None:
        best = _init_seeds()[0]
        best_side = _outer_side(_verts(best))
    # Final corrective pass to guarantee a real gap
    if _elapsed() < DEADLINE - 5:
        vl = _verts(best)
        if _max_pen(vl) > -1e-4:
            res = minimize(_obj_hard, best, args=(1e6,), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True})
            vl2 = _verts(res.x)
            if _max_pen(vl2) <= _max_pen(vl):
                best = res.x
                best_side = _outer_side(vl2)
    inner = [[float(best[3*i]), float(best[3*i+1]), float(best[3*i+2])] for i in range(N)]
    return inner, (0.0, 0.0), float(_outer_side(_verts(best))), 0.0
