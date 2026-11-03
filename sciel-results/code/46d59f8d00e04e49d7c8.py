import time
import numpy as np

N = 12
_DEADLINE = time.time() + 140.0

def _hex_verts(cx, cy, ang_deg, side=1.0):
    a = np.deg2rad(ang_deg)
    base = np.array([[np.cos(np.deg2rad(60*i)), np.sin(np.deg2rad(60*i))] for i in range(6)])
    R = np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]])
    return base @ R.T + np.array([cx, cy])

_AXES = np.array([[np.cos(np.deg2rad(60*i)), np.sin(np.deg2rad(60*i))] for i in range(3)])

def _outer_side(hexagons):
    r = 0.0
    for verts in hexagons:
        for ax in _AXES:
            r = max(r, np.abs(verts @ ax).max())
    return 2.0 * r / np.sqrt(3.0)

def _penetration(v1, v2):
    best = 0.0
    for p in (v1, v2):
        for i in range(6):
            a = p[i]; b = p[(i+1)%6]
            edge = b - a
            n = np.array([-edge[1], edge[0]])
            nn = np.linalg.norm(n)
            if nn < 1e-12: continue
            n = n / nn
            p1 = v1 @ n; p2 = v2 @ n
            sep = min(p1.min() - p2.max(), p2.min() - p1.max())
            if sep > best: best = sep
    return -best

def _build(params):
    return [_hex_verts(params[3*k], params[3*k+1], params[3*k+2]) for k in range(N)]

def _objective(params, pen=50.0):
    hexagons = _build(params)
    s = _outer_side(hexagons)
    maxp = 0.0
    for i in range(N):
        for j in range(i+1, N):
            p = _penetration(hexagons[i], hexagons[j])
            if p > maxp: maxp = p
    return s + pen * maxp

def _inits():
    inits = []
    # ring
    r1 = np.zeros(3*N)
    for k in range(N):
        ang = 2*np.pi*k/N
        r1[3*k] = 2.6*np.cos(ang); r1[3*k+1] = 2.6*np.sin(ang); r1[3*k+2] = 0.0
    inits.append(r1)
    # grid 4x3
    g = np.zeros(3*N)
    pos = [(x*2.0, y*2.0) for y in range(3) for x in range(4)]
    for k in range(N):
        g[3*k], g[3*k+1] = pos[k]; g[3*k+2] = 0.0
    inits.append(g)
    return inits

def construct():
    from scipy.optimize import minimize
    best_params = None; best_val = np.inf
    for init in _inits():
        if time.time() > _DEADLINE: break
        try:
            res = minimize(_objective, init, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol':1e-4, 'fatol':1e-5})
            if res.fun < best_val:
                best_val = res.fun; best_params = res.x
        except Exception:
            pass
    if best_params is None:
        best_params = _inits()[0]
    inner = [[float(best_params[3*k]), float(best_params[3*k+1]), float(best_params[3*k+2])] for k in range(N)]
    side = float(_outer_side(_build(best_params)))
    return inner, (0.0, 0.0), side, 0.0
