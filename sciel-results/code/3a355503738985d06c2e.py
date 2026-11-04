import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    deadline = time.time() + 140.0

    # ---------- geometry helpers ----------
    def _hex_verts(cx, cy, a_deg):
        a = math.radians(a_deg)
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, -s], [s, c]])
        return np.array([[math.cos(i*math.pi/3), math.sin(i*math.pi/3)] for i in range(6)]) @ R.T + np.array([cx, cy])

    def sat_disjoint(h1, h2, eps=1e-6):
        v1 = _hex_verts(h1[0], h1[1], h1[2])
        v2 = _hex_verts(h2[0], h2[1], h2[2])
        for v in (v1, v2):
            for i in range(6):
                dx = v[(i+1)%6,0] - v[i,0]
                dy = v[(i+1)%6,1] - v[i,1]
                ln = math.hypot(dx, dy)
                if ln < 1e-10:
                    continue
                n = np.array([-dy/ln, dx/ln])
                p1 = v1 @ n
                p2 = v2 @ n
                if p1.max() < p2.min() - eps or p2.max() < p1.min() - eps:
                    return True
        return False

    def all_disjoint(inner, eps=1e-6):
        for i in range(N):
            for j in range(i+1, N):
                if not sat_disjoint(inner[i], inner[j], eps):
                    return False
        return True

    # ---------- objective (smooth epigraph surrogate) ----------
    def outer_side(inner, cx, cy, theta_deg, beta=200.0):
        theta = math.radians(theta_deg)
        mx = -1e9
        for k in range(6):
            na = theta + math.pi/6 + k*math.pi/3
            nx, ny = math.cos(na), math.sin(na)
            m = -1e9
            for x, y, a in inner:
                v = _hex_verts(x, y, a)
                pr = (v[:,0]-cx)*nx + (v[:,1]-cy)*ny
                m = max(m, pr.max())
            mx = max(mx, m)
        if mx <= 0:
            mx = 0.0
        return 2.0 * (mx + math.log(1.0 + math.exp(-beta*mx))/beta) / math.sqrt(3)

    def objective(x, beta=200.0):
        inner = [[x[3*i], x[3*i+1], x[3*i+2]] for i in range(N)]
        cx, cy, th = x[3*N], x[3*N+1], x[3*N+2]
        for i in range(N):
            for j in range(i+1, N):
                if not sat_disjoint(inner[i], inner[j], 1e-7):
                    return 1e6
        return outer_side(inner, cx, cy, th, beta)

    # ---------- exact lattice topologies ----------
    def lattice_topology():
        d = math.sqrt(3) + 0.01
        u = np.array([d, 0.0])
        v = np.array([d/2.0, d*math.sqrt(3)/2.0])
        centers = []
        for r in range(3):
            for c in range(4):
                centers.append(r*v + c*u)
        return [[c[0], c[1], 30.0] for c in centers]

    def ring_topology():
        d = math.sqrt(3) + 0.01
        u = np.array([d, 0.0])
        v = np.array([d/2.0, d*math.sqrt(3)/2.0])
        centers = [np.array([0.0, 0.0])]
        r1 = [u, v, v-u, -u, -v, u-v]
        for vec in r1:
            centers.append(vec.copy())
        cands = []
        for p1 in r1:
            for b in r1:
                p2 = p1 + b
                if np.allclose(p2, np.array([0.0,0.0])):
                    continue
                if any(np.allclose(p2, q) for q in r1):
                    continue
                if any(np.allclose(p2, q) for q in cands):
                    continue
                cands.append(p2)
        cands.sort(key=lambda p: np.linalg.norm(p))
        for i in range(5):
            centers.append(cands[i].copy())
        return [[c[0], c[1], 30.0] for c in centers]

    def evaluate(inner):
        verts = []
        for x, y, a in inner:
            verts.extend(_hex_verts(x, y, a))
        verts = np.array(verts)
        cx, cy = verts[:,0].mean(), verts[:,1].mean()
        best, ba = float('inf'), 0.0
        for th in np.linspace(0, 60, 121):
            s = outer_side(inner, cx, cy, th)
            if s < best:
                best, ba = s, th
        return inner, cx, cy, ba, best

    topologies = [lattice_topology(), ring_topology()]
    best = None
    best_side = float('inf')
    for t in topologies:
        if time.time() > deadline - 25.0:
            break
        if not all_disjoint(t):
            continue
        inner, cx, cy, ba, bs = evaluate(t)
        if bs < best_side:
            best = (inner, cx, cy, ba, bs)
            best_side = bs
    if best is None:
        best = evaluate(lattice_topology())
        best_side = best[4]
    inner, cx, cy, ba, best_side = best

    # ---------- smooth local polish ----------
    def polish(inner, cx, cy, th, beta=200.0, maxiter=300):
        x0 = np.array([v for h in inner for v in h] + [cx, cy, th])
        try:
            r = minimize(objective, x0, args=(beta,), method='Nelder-Mead',
                         options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True})
            xo = r.x
            io = [[xo[3*i], xo[3*i+1], xo[3*i+2]] for i in range(N)]
            cX, cY, tH = xo[3*N], xo[3*N+1], xo[3*N+2]
            if all_disjoint(io, 1e-7):
                s = outer_side(io, cX, cY, tH)
                if s < best_side:
                    return io, cX, cY, tH, s
        except Exception:
            pass
        return inner, cx, cy, th, best_side

    if time.time() < deadline - 20.0:
        inner, cx, cy, ba, best_side = polish(inner, cx, cy, ba)

    # ---------- high-precision coordinate polish ----------
    if time.time() < deadline - 3.0:
        step = 0.002
        for _ in range(200):
            if time.time() > deadline - 2.0:
                break
            improved = False
            for i in range(N):
                for dx in (-step, step):
                    for dy in (-step, step):
                        ni = [list(h) for h in inner]
                        ni[i][0] += dx
                        ni[i][1] += dy
                        if not all_disjoint(ni):
                            continue
                        ns = outer_side(ni, cx, cy, ba)
                        if ns < best_side - 1e-12:
                            inner = ni
                            best_side = ns
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                step *= 0.7
                if step < 1e-5:
                    break

    outer_center = (cx, cy)
    outer_side_val = best_side
    outer_angle = ba

    # ---------- final safety ----------
    if not all_disjoint(inner, 1e-5):
        inner = lattice_topology()
        verts = []
        for x, y, a in inner:
            verts.extend(_hex_verts(x, y, a))
        verts = np.array(verts)
        cx, cy = verts[:,0].mean(), verts[:,1].mean()
        bs, ba = float('inf'), 0.0
        for th in np.linspace(0, 60, 121):
            s = outer_side(inner, cx, cy, th)
            if s < bs:
                bs, ba = s, th
        outer_center = (cx, cy)
        outer_side_val = bs
        outer_angle = ba

    return inner, outer_center, outer_side_val, outer_angle
