"""Hexagon packing via support-function LP + Nelder-Mead polish."""
import time
import math
import numpy as np
from scipy.optimize import linprog, minimize

N = 12
DEADLINE = 150.0
START = time.time()

def _hex_verts(cx, cy, side, ang_deg):
    a = math.radians(ang_deg)
    v = []
    for k in range(6):
        th = a + k * math.pi / 3.0
        v.append((cx + side * math.cos(th), cy + side * math.sin(th)))
    return np.array(v)

def _support(v, ux, uy):
    return float(np.max(v * np.array([ux, uy])))

def _separation(A, B):
    # min over directions of (sup_B(u) - sup_A(u))  (>=0 iff disjoint, touching=0)
    # A,B are vertex arrays (6,2)
    best = float('inf')
    for ux, uy in _DIRS:
        sA = float(np.max(A * np.array([ux, uy])))
        sB = float(np.max(B * np.array([ux, uy])))
        d = sB - sA
        if d < best:
            best = d
    return best

# Dense directional sampling for Minkowski-difference support
_NSAMP = 360
_ANGS = np.linspace(0.0, 2.0 * math.pi, _NSAMP, endpoint=False)
_DIRS = [(math.cos(a), math.sin(a)) for a in _ANGS]

def _outer_side_for(inner, center, outer_ang_deg):
    # Minimal outer side s (with given center+angle) such that all inner hexes are inside.
    # Inner hex i has support h_i(u) = sup over its verts.
    # Outer hexagon centered at c, side s, angle o: support in direction u is
    #   s * |u . e1| + s * |u . e2|  where e1,e2 are two adjacent edge directions
    #   (support function of a regular hexagon of side s).
    o = math.radians(outer_ang_deg)
    e1 = np.array([math.cos(o), math.sin(o)])
    e2 = np.array([math.cos(o + math.pi / 3.0), math.sin(o + math.pi / 3.0)])
    need = 0.0
    for (cx, cy, ang) in inner:
        v = _hex_verts(cx, cy, 1.0, ang)
        # translate so outer center is origin
        v2 = v - np.array(center)
        for ux, uy in _DIRS:
            u = np.array([ux, uy])
            s_inner = float(np.max(v2 * u))
            # outer support coefficient in direction u:
            coef = abs(float(np.dot(u, e1))) + abs(float(np.dot(u, e2)))
            if coef > 1e-12:
                req = s_inner / coef
                if req > need:
                    need = req
    return max(need, 1e-9)

def _objective(x):
    # x layout: [cx_i, cy_i, ang_i]*N + [ocx, ocy, oang]
    n = len(x)
    inner = []
    for i in range(N):
        inner.append((x[3*i], x[3*i+1], x[3*i+2]))
    ocx, ocy, oang = x[-3], x[-2], x[-1]
    # non-overlap penalty
    pen = 0.0
    for i in range(N):
        vi = _hex_verts(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for j in range(i+1, N):
            vj = _hex_verts(inner[j][0], inner[j][1], 1.0, inner[j][2])
            sep = _separation(vi, vj)
            if sep < 0:
                pen += (-sep) ** 2
    s = _outer_side_for(inner, (ocx, ocy), oang)
    return s + 100.0 * pen

def _initial():
    # 1+6+5 cluster: center + ring of 6 + 5 around
    inner = []
    # center
    inner.append((0.0, 0.0, 0.0))
    # ring of 6 at distance ~2 (side 1 hexes touch at distance 2 along flat directions)
    for k in range(6):
        a = math.radians(60 * k)
        inner.append((2.0 * math.cos(a), 2.0 * math.sin(a), 0.0))
    # 5 more in a second ring (partial)
    for k in range(5):
        a = math.radians(60 * k + 30)
        inner.append((3.464 * math.cos(a), 3.464 * math.sin(a), 0.0))
    x = []
    for (cx, cy, ang) in inner:
        x += [cx, cy, ang]
    x += [0.0, 0.0, 0.0]
    return x

def construct():
    global START
    t0 = time.time()
    if START is None:
        START = t0

    best_x = _initial()
    best_f = _objective(best_x)

    # Nelder-Mead polish
    opts = dict(xatol=1e-8, fatol=1e-10, maxiter=4000, maxfev=4000)
    try:
        res = minimize(_objective, best_x, method='Nelder-Mead', options=opts)
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
    except Exception:
        pass

    if time.time() - START > DEADLINE - 20:
        pass
    else:
        # second polish with tighter tolerance
        opts2 = dict(xatol=1e-10, fatol=1e-12, maxiter=3000, maxfev=3000)
        try:
            res2 = minimize(_objective, best_x, method='Nelder-Mead', options=opts2)
            if res2.fun < best_f:
                best_f = res2.fun
                best_x = res2.x
        except Exception:
            pass

    inner = []
    for i in range(N):
        inner.append([best_x[3*i], best_x[3*i+1], best_x[3*i+2]])
    ocx, ocy, oang = best_x[-3], best_x[-2], best_x[-1]
    side = _outer_side_for(inner, (ocx, ocy), oang)
    return inner, (ocx, ocy), side, oang
