import numpy as np
import time
from scipy.optimize import minimize

N = 12

def _hex_vertices(cx, cy, s, angle_deg):
    a = np.radians(angle_deg)
    return np.array([[cx + s * np.cos(a + i * np.pi / 3),
                      cy + s * np.sin(a + i * np.pi / 3)] for i in range(6)])

def _min_enclosing_hexagon(inner_positions, inner_angles):
    all_verts = np.vstack([_hex_vertices(x, y, 1.0, a) for x, y, a in inner_positions])
    best_side = float('inf')
    best_center = None
    best_angle = 0.0
    for phi_deg in np.linspace(0, 60, 360):
        phi = np.radians(phi_deg)
        normals = np.array([[np.cos(phi + i * np.pi / 3 + np.pi / 6),
                             np.sin(phi + i * np.pi / 3 + np.pi / 6)] for i in range(6)])
        projs = all_verts @ normals.T
        max_p, min_p = projs.max(axis=0), projs.min(axis=0)
        apothem = np.max(max_p - min_p) / 2
        s_cand = apothem * 2 / np.sqrt(3)
        if s_cand >= best_side:
            continue
        lo, hi = max_p - apothem, min_p + apothem
        if np.any(lo > hi + 1e-12):
            continue
        mid = (lo + hi) / 2
        c, _, _, _ = np.linalg.lstsq(normals, mid, rcond=None)
        if np.any(normals @ c < lo - 1e-9) or np.any(normals @ c > hi + 1e-9):
            continue
        best_side, best_center, best_angle = s_cand, c, phi_deg
    if best_center is None:
        centroid = all_verts.mean(axis=0)
        return centroid[0], centroid[1], 100.0, 0.0
    return best_center[0], best_center[1], best_side, best_angle

def _sat_overlap_depth(h1, h2):
    min_depth = float('inf')
    axes = []
    for poly in [h1, h2]:
        n = len(poly)
        for i in range(n):
            edge = poly[(i+1)%n] - poly[i]
            norm = np.linalg.norm(edge)
            if norm < 1e-10: continue
            axes.append(np.array([-edge[1], edge[0]]) / norm)
    for axis in axes:
        p1, p2 = h1 @ axis, h2 @ axis
        if p1.max() <= p2.min() or p2.max() <= p1.min():
            return 0.0
        min_depth = min(min_depth, min(p1.max(), p2.max()) - max(p1.min(), p2.min()))
    return min_depth

def _penalty(params, outer_side, outer_center, outer_angle):
    iv = [_hex_vertices(params[3*i], params[3*i+1], 1.0, params[3*i+2]) for i in range(N)]
    ov = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    pen = 0.0
    for v in iv:
        for i in range(6):
            v1, v2 = ov[i], ov[(i+1)%6]
            e = v2 - v1
            el = np.linalg.norm(e)
            if el < 1e-12: continue
            for pt in v:
                rel = pt - v1
                cross = e[0]*rel[1] - e[1]*rel[0]
                if cross < 0:
                    pen += (-cross) / el
    for i in range(N):
        for j in range(i+1, N):
            d = _sat_overlap_depth(iv[i], iv[j])
            if d > 0: pen += d * 10.0
    return pen

def _honeycomb(count, pattern, rot):
    s3 = np.sqrt(3)
    sc = 1.001
    pos = []
    for j, rl in enumerate(pattern):
        for i in range(rl):
            pos.append(((i + 0.5*(j%2))*s3*sc, j*1.5*sc, rot))
    return pos[:count]

def construct():
    t0 = time.time()
    dl = t0 + 140.0
    best = float('inf')
    best_r = None
    
    pats = [[3,4,3,2],[4,3,3,2],[2,3,4,3],[3,3,4,2],[4,4,2,2],[2,2,4,4],
            [5,3,2,2],[2,2,3,5],[6,3,3],[3,3,6],[4,4,4],[2,4,4,2],
            [4,2,4,2],[2,4,2,4],[3,5,4],[4,5,3],[5,4,3],[3,4,5],
            [2,5,5],[5,5,2],[5,2,5],[1,4,4,3],[3,4,4,1]]
    
    for p in pats:
        if sum(p) != 12 or len(p) > 6: continue
        if time.time() > dl - 100: break
        for rot in [0, 15, 30, 45, 60]:
            if time.time() > dl - 100: break
            pos = _honeycomb(12, p, rot)
            if len(pos) != 12: continue
            cx, cy, s, a = _min_enclosing_hexagon(pos, [r[2] for r in pos])
            ss = s * 1.0001 + 1e-6
            if ss < best:
                best = ss
                best_r = (pos, (cx, cy), ss, a)
    
    if best_r is None:
        return [(4*k, 0, 0) for k in range(-6, 6)], (0, 0), 40.0, 0.0
    
    inner0, (cx0, cy0), s0, a0 = best_r
    p0 = np.array([v for x, y, a in inner0 for v in [x, y, a]])
    
    configs = [(a0, s0*0.99), (a0, s0*0.97), (a0+10, s0*0.99), (a0-10, s0*0.99)]
    
    for oa, os_ in configs:
        if time.time() > dl - 20: break
        oc = (cx0, cy0)
        def obj(p): return _penalty(p, os_, oc, oa)
        try:
            r = minimize(obj, p0, method='L-BFGS-B',
                         options={'maxiter': 300, 'maxfun': 8000, 'ftol': 1e-14, 'gtol': 1e-10})
            if r.fun < 1e-5:
                io = [(r.x[3*i], r.x[3*i+1], r.x[3*i+2]) for i in range(N)]
                cxo, cyo, so, ao = _min_enclosing_hexagon(io, [x[2] for x in io])
                sso = so * 1.0001 + 1e-6
                if sso < best:
                    best = sso
                    best_r = (io, (cxo, cyo), sso, ao)
        except: pass
    
    if best_r is None:
        return [(4*k, 0, 0) for k in range(-6, 6)], (0, 0), 40.0, 0.0
    
    return best_r
