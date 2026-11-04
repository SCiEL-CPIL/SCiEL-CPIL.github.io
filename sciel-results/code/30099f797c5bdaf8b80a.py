"""Packing 12 unit hexagons in a minimal regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
START_TIME = time.time()
TIME_LIMIT = 130.0

def _hex_vertices(cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    return np.array([[cx + side * np.cos(angle + i * np.pi / 3),
                      cy + side * np.sin(angle + i * np.pi / 3)] for i in range(6)])

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    dx, dy = px - cx, py - cy
    apothem = side * np.sqrt(3) / 2
    for i in range(6):
        ea = angle + i * np.pi / 3 + np.pi / 6
        proj = dx * np.cos(ea) + dy * np.sin(ea)
        if proj > apothem + 1e-9:
            return False
    return True

def _sat_overlap(v1, v2):
    """Returns True if two hexagons (as vertex arrays) overlap, using SAT."""
    axes = []
    for v in (v1, v2):
        for k in range(6):
            e = v[(k + 1) % 6] - v[k]
            n = np.array([-e[1], e[0]])
            nn = np.linalg.norm(n)
            if nn > 1e-12:
                axes.append(n / nn)
    for ax in axes:
        p1 = v1 @ ax
        p2 = v2 @ ax
        if np.max(p1) < np.min(p2) or np.max(p2) < np.min(p1):
            return False
    return True

def _overlap_penalty(inner_params):
    n = N
    verts = []
    for i in range(n):
        x, y, a = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
        verts.append(_hex_vertices(x, y, 1.0, a))
    pen = 0.0
    for i in range(n):
        for j in range(i+1, n):
            if _sat_overlap(verts[i], verts[j]):
                # Approximate penetration
                d = np.linalg.norm(verts[i][:,0].mean() - verts[j][:,0].mean())
                pen += max(0, 2.0 - d) ** 2
    return pen

def _containment_penalty(inner_params, outer_params):
    ox, oy, os, oa = outer_params
    pen = 0.0
    for i in range(N):
        x, y, a = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
        v = _hex_vertices(x, y, 1.0, a)
        for pt in v:
            if not _point_in_hexagon(pt[0], pt[1], ox, oy, os, oa):
                dx, dy = pt[0]-ox, pt[1]-oy
                ang = np.radians(oa)
                apothem = os * np.sqrt(3) / 2
                for k in range(6):
                    ea = ang + k*np.pi/3 + np.pi/6
                    proj = dx*np.cos(ea) + dy*np.sin(ea)
                    if proj > apothem:
                        pen += (proj - apothem)**2
    return pen

def _objective(params):
    ip = params[:3*N]
    op = params[3*N:3*N+4]
    os = op[2]
    if os < 2.0:
        return 1000.0 + (2.0-os)**2
    return os + 10.0*_overlap_penalty(ip) + 10.0*_containment_penalty(ip, op)

def _honeycomb(rows):
    pos = []
    y = 0.0
    for cnt in rows:
        x0 = -(cnt-1)*np.sqrt(3)/2
        for k in range(cnt):
            pos.append([x0 + k*np.sqrt(3), y, 0.0])
        y += 1.5
    return pos

# EVOLVE-BLOCK-START
def construct():
    global START_TIME
    START_TIME = time.time()
    configs = [[1,2,3,3,2,1],[2,3,4,3],[3,4,3,2],[4,4,4],[3,3,3,3],
               [2,4,4,2],[1,3,4,4,3,1],[2,2,4,4],[3,6,3],[1,2,3,4,2]]
    best = None
    best_sc = float('inf')
    for rows in configs:
        if sum(rows) != N:
            continue
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        init = _honeycomb(rows)
        xs = [p[0] for p in init]
        ys = [p[1] for p in init]
        ox = (max(xs)+min(xs))/2
        oy = (max(ys)+min(ys))/2
        os = max((max(xs)-min(xs))/np.sqrt(3), (max(ys)-min(ys))/1.5) + 2.0
        os = max(os, 4.0)
        params = []
        for p in init:
            params.extend(p)
        params.extend([ox, oy, os, 0.0])
        try:
            res = minimize(_objective, params, method='Nelder-Mead',
                           options={'maxiter':1500,'xatol':1e-5,'fatol':1e-5,'adaptive':True})
            fp = res.x
            os = fp[3*N+2]
            if os < best_sc and os > 2.0:
                inner = [[fp[3*i], fp[3*i+1], fp[3*i+2]] for i in range(N)]
                best = (inner, (fp[3*N], fp[3*N+1]), os, fp[3*N+3])
                best_sc = os
        except Exception:
            pass
    if best is not None:
        return best
    inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N-N//2)]
    return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END
