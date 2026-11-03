import math
import time
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135.0  # 150s limit, keep margin
    sqrt3 = math.sqrt(3.0)

    # ---------- geometry helpers ----------
    def hex_verts(cx, cy, side, ang):
        a = math.radians(ang)
        ca, sa = math.cos(a), math.sin(a)
        base = np.array([[1.0, 0.0], [0.5, sqrt3/2], [-0.5, sqrt3/2],
                         [-1.0, 0.0], [-0.5, -sqrt3/2], [0.5, -sqrt3/2]]) * side
        rot = np.array([[ca, -sa], [sa, ca]])
        return (base @ rot.T) + np.array([cx, cy])

    def hex_halfplanes(cx, cy, side, ang):
        # returns (A, b) such that A p <= b  (A: 6x2, b: 6)
        a = math.radians(ang)
        ca, sa = math.cos(a), math.sin(a)
        n = (np.pi/6) + np.arange(6) * (np.pi/3) + a
        A = np.stack([np.cos(n), np.sin(n)], axis=1)
        b = np.full(6, side * sqrt3 / 2.0)
        return A, b, np.array([cx, cy])

    def overlap_penalty(A, b, B, c, cc):
        # max over edges of A of (A@cc - b) + (B@cc - c)
        return np.max(A @ cc - b) + np.max(B @ cc - c)

    def feasible(inner, tol=1e-7):
        # inner: (N,3) [x,y,ang_deg]
        n = inner.shape[0]
        hp = []
        for k in range(n):
            A, b, cc = hex_halfplanes(inner[k,0], inner[k,1], 1.0, inner[k,2])
            hp.append((A, b, cc))
        for i in range(n):
            for j in range(i+1, n):
                A, b, ci = hp[i]
                B, c, cj = hp[j]
                cc = 0.5*(ci + cj)
                if overlap_penalty(A, b, B, c, cc) > tol:
                    return False
        return True

    def outer_side_and_center(inner, cx, cy, ang_deg):
        # compute required side length for outer hexagon with given center/rotation
        a = math.radians(ang_deg)
        n = (np.pi/6) + np.arange(6)*(np.pi/3) + a
        Nrm = np.stack([np.cos(n), np.sin(n)], axis=1)  # 6x2
        verts = np.vstack([hex_verts(x, y, 1.0, ang) for (x, y, ang) in inner])
        rel = verts - np.array([cx, cy])
        proj = rel @ Nrm.T  # (V,6)
        maxproj = np.max(proj, axis=0)
        s = 2.0 * np.max(maxproj) / sqrt3
        return s

    # ---------- build initial lattice cluster (7 + 5) ----------
    d = sqrt3  # center distance for touching unit hexagons
    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*sqrt3/2.0])
    basis = [u, v, v - u, -u, -v, u - v]

    centers = [np.array([0.0, 0.0])]
    ring1 = [b.copy() for b in basis]
    centers.extend(ring1)

    ring2 = []
    for p1 in ring1:
        for b in basis:
            p2 = p1 + b
            if np.allclose(p2, np.array([0.0, 0.0]), atol=1e-6):
                continue
            if any(np.allclose(p2, r, atol=1e-6) for r in ring1):
                continue
            if any(np.allclose(p2, q, atol=1e-6) for q in ring2):
                continue
            ring2.append(p2.copy())
    ring2.sort(key=lambda p: np.linalg.norm(p))
    for p in ring2[:5]:
        centers.append(p)
    centers = centers[:N]

    inner0 = np.array([[c[0], c[1], 30.0] for c in centers], dtype=float)

    # ---------- objective for optimizer ----------
    def objective(x):
        # x: N*3 + 3 (outer cx, cy, ang)
        inner = x[:N*3].reshape(N, 3)
        ocx, ocy, oang = x[N*3], x[N*3+1], x[N*3+2]

        # outer side length
        a = math.radians(oang)
        n = (np.pi/6) + np.arange(6)*(np.pi/3) + a
        Nrm = np.stack([np.cos(n), np.sin(n)], axis=1)
        verts = np.vstack([hex_verts(px, py, 1.0, pa) for (px, py, pa) in inner])
        rel = verts - np.array([ocx, ocy])
        maxproj = np.max(rel @ Nrm.T, axis=0)
        s = 2.0 * np.max(maxproj) / sqrt3

        # overlap penalty (cheap, using midpoint)
        hp = []
        for k in range(N):
            A, b, cc = hex_halfplanes(inner[k,0], inner[k,1], 1.0, inner[k,2])
            hp.append((A, b, cc))
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                A, b, ci = hp[i]
                B, c, cj = hp[j]
                cc = 0.5*(ci + cj)
                op = overlap_penalty(A, b, B, c, cc)
                if op > 0:
                    pen += op
        return s + 1000.0 * pen

    # ---------- try scipy SLSQP from lattice start ----------
    def try_slsqp(start, max_iter=200, time_budget=120.0):
        x0 = start.copy()
        try:
            from scipy.optimize import minimize
            t0 = time.time()
            res = minimize(objective, x0, method='SLSQP',
                           options={'maxiter': max_iter, 'ftol': 1e-9})
            if time.time() - t0 > time_budget:
                return None
            x = res.x
            inner = x[:N*3].reshape(N, 3)
            ocx, ocy, oang = x[N*3], x[N*3+1], x[N*3+2]
            if not feasible(inner):
                return None
            s = outer_side_and_center(inner, ocx, ocy, oang)
            return s, (float(ocx), float(ocy)), float(oang), inner
        except Exception:
            return None

    # best known so far
    inner_best = inner0.copy()
    # compute a good outer for the lattice start
    cx0 = float(np.mean(inner0[:, 0]))
    cy0 = float(np.mean(inner0[:, 1]))
    # find best outer angle
    best_s = float('inf')
    best_ang = 0.0
    for th in np.linspace(0, 60, 121):
        s = outer_side_and_center(inner0, cx0, cy0, th)
        if s < best_s:
            best_s = s
            best_ang = float(th)
    outer_center = (cx0, cy0)
    outer_side = best_s
    outer_angle = best_ang

    # SLSQP from the lattice start
    x0 = np.concatenate([inner0.flatten(), [cx0, cy0, best_ang]])
    if time.time() < deadline - 15:
        r = try_slsqp(x0, max_iter=300, time_budget=deadline - time.time() - 10)
        if r is not None and r[0] < outer_side - 1e-7:
            outer_side, outer_center, outer_angle, inner_best = r

    # ---------- additional random restarts with local SLSQP ----------
    if time.time() < deadline - 10:
        for restart in range(8):
            if time.time() > deadline - 8:
                break
            # perturb from best
            inner_r = inner_best.copy()
            inner_r[:, 0] += random.uniform(-0.5, 0.5)
            inner_r[:, 1] += random.uniform(-0.5, 0.5)
            inner_r[:, 2] += random.uniform(-20, 20)
            cxr = outer_center[0] + random.uniform(-0.5, 0.5)
            cyr = outer_center[1] + random.uniform(-0.5, 0.5)
            ar = outer_angle + random.uniform(-15, 15)
            xr = np.concatenate([inner_r.flatten(), [cxr, cyr, ar]])
            r = try_slsqp(xr, max_iter=150, time_budget=min(20.0, deadline - time.time() - 5))
            if r is not None and r[0] < outer_side - 1e-7:
                outer_side, outer_center, outer_angle, inner_best = r

    # ---------- final local polish: fine outer angle + center ----------
    if time.time() < deadline - 3:
        cx = outer_center[0]
        cy = outer_center[1]
        for th in np.linspace(0, 60, 361):
            s = outer_side_and_center(inner_best, cx, cy, th)
            if s < outer_side - 1e-9:
                outer_side = s
                outer_angle = float(th)
        # refine center via small grid
        for dcx in np.linspace(-0.3, 0.3, 7):
            for dcy in np.linspace(-0.3, 0.3, 7):
                s = outer_side_and_center(inner_best, cx+dcx, cy+dcy, outer_angle)
                if s < outer_side - 1e-9:
                    outer_side = s
                    outer_center = (cx+dcx, cy+dcy)

    # ---------- build return ----------
    inner_list = [[float(x), float(y), float(a)] for (x, y, a) in inner_best]
    return inner_list, outer_center, float(outer_side), float(outer_angle)
# EVOLVE-BLOCK-END
