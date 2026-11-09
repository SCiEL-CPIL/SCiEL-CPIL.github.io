"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_T0 = time.time()
_DEADLINE = 145.0

def _hex_verts(cx, cy, s, rot):
    vs = np.empty((6, 2))
    for k in range(6):
        a = math.radians(60.0 * k) + rot
        vs[k, 0] = cx + s * math.cos(a)
        vs[k, 1] = cy + s * math.sin(a)
    return vs

def _point_in_hex(px, py, cx, cy, s, rot):
    dx = px - cx
    dy = py - cy
    ca, sa = math.cos(-rot), math.sin(-rot)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    r_in = s * math.cos(math.radians(30))
    c30 = math.cos(math.radians(30))
    d1 = abs(c30 * lx + 0.5 * ly)
    d2 = abs(ly)
    d3 = abs(-c30 * lx + 0.5 * ly)
    return max(d1, d2, d3) <= r_in + 1e-9

def _build_objective(n):
    def objective(p):
        x = p[:2*n]
        angles = p[2*n:3*n]
        t = p[3*n]
        pen = 0.0
        for i in range(n):
            xi, yi = x[2*i], x[2*i+1]
            ai = angles[i]
            for k in range(6):
                a = math.radians(60.0*k) + ai
                vx = xi + math.cos(a)
                vy = yi + math.sin(a)
                if not _point_in_hex(vx, vy, 0.0, 0.0, t, 0.0):
                    pen += 10.0
        for i in range(n):
            vi = _hex_verts(x[2*i], x[2*i+1], 1.0, angles[i])
            for j in range(i+1, n):
                vj = _hex_verts(x[2*j], x[2*j+1], 1.0, angles[j])
                dc = math.sqrt((x[2*i]-x[2*j])**2 + (x[2*i+1]-x[2*j+1])**2)
                if dc < 4.0:
                    for m in range(6):
                        if _point_in_hex(vi[m,0], vi[m,1], x[2*j], x[2*j+1], 1.0, angles[j]):
                            pen += 10.0
                            break
                        if _point_in_hex(vj[m,0], vj[m,1], x[2*i], x[2*i+1], 1.0, angles[i]):
                            pen += 10.0
                            break
        return 0.5*t*t + pen
    return objective

def construct():
    t_start = time.time()
    n = N

    def _remaining():
        return _DEADLINE - (time.time() - t_start)

    objective = _build_objective(n)
    best_val = float('inf')
    best_p = None

    rng = np.random.RandomState(123)

    for trial in range(8):
        if _remaining() < 15.0:
            break

        p = np.zeros(3*n + 1)
        p[0] = 0.0; p[1] = 0.0
        for k in range(6):
            a = math.radians(60.0*k)
            p[2*(1+k)] = 2.0*math.cos(a)
            p[2*(1+k)+1] = 2.0*math.sin(a)
        for k in range(4):
            a = math.radians(60.0*k + 30.0)
            p[2*(7+k)] = 4.0*math.cos(a)
            p[2*(7+k)+1] = 4.0*math.sin(a)
        if trial > 0:
            p[:3*n] += rng.randn(3*n) * 0.4
        p[3*n] = 7.0

        try:
            res = minimize(objective, p, method='Nelder-Mead',
                          options={'maxiter': 400, 'xatol': 1e-3, 'fatol': 1e-4})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x.copy()
        except Exception:
            pass

    if best_p is None:
        inner = [[float(k*2.0), 0.0, 0.0] for k in range(n)]
        return inner, (0.0, 0.0), 40.0, 0.0

    x = best_p[:2*n]
    angles = best_p[2*n:3*n]
    t = best_p[3*n]

    inner = [[float(x[2*i]), float(x[2*i+1]), float(math.degrees(angles[i]))] for i in range(n)]

    if t < 3.5:
        inner = [[float(k*2.0), 0.0, 0.0] for k in range(n)]
        t = 40.0

    return inner, (0.0, 0.0), float(t), 0.0
