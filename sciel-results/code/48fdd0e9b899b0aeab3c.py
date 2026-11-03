"""Ring placement of 12 unit hexagons, optimized with Nelder-Mead."""
import math
import time

import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 120.0
_EPS = 1e-9


def _hex_verts(cx, cy, s, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    vx = [s * math.cos(math.radians(60.0 * i)) for i in range(6)]
    vy = [s * math.sin(math.radians(60.0 * i)) for i in range(6)]
    rx = [ca * x - sa * y for x, y in zip(vx, vy)]
    ry = [sa * x + ca * y for x, y in zip(vx, vy)]
    return np.array([(rx[i] + cx, ry[i] + cy) for i in range(6)])


def _sat_overlap(A, B, tol=1e-9):
    """Separating Axis Theorem: True if convex polygons A,B overlap (interior)."""
    polys = (A, B)
    for P in polys:
        for i in range(len(P)):
            ax = P[i + 1][0] - P[i][0]
            ay = P[i + 1][1] - P[i][1]
            # normal (perpendicular)
            nx, ny = -ay, ax
            nlen = math.hypot(nx, ny)
            if nlen < 1e-12:
                continue
            nx /= nlen
            ny /= nlen
            minA = min(p[0] * nx + p[1] * ny for p in A)
            maxA = max(p[0] * nx + p[1] * ny for p in A)
            minB = min(p[0] * nx + p[1] * ny for p in B)
            maxB = max(p[0] * nx + p[1] * ny for p in B)
            if maxA < minB + tol or maxB < minA + tol:
                return False
    return True


def _outer_side_from_points(pts, rot_deg):
    """Min side length of a regular hexagon (center 0, rotation rot) containing pts."""
    a = math.radians(rot_deg)
    ca, sa = math.cos(a), math.sin(a)
    # outward normals of the 6 sides (pointing outward)
    normals = []
    for i in range(6):
        th = math.radians(60.0 * i)
        # side i is perpendicular to direction th, outward normal = (cos th, sin th)
        normals.append((math.cos(th), math.sin(th)))
    # In a regular hexagon centered at origin, support in direction n is s*|n·u|
    # where u is the direction to the vertex. We need s such that for every point p,
    # for every outward normal n: p·n <= s * (n·u) where u is the vertex direction.
    # For a regular hexagon, the support function h(n) = s * max_k (n · v_k) where v_k are
    # unit vertices. We compute required s = max over points and normals of p·n / h_unit(n).
    unit_verts = [np.array([math.cos(math.radians(60.0 * i)),
                            math.sin(math.radians(60.0 * i))]) for i in range(6)]
    # transform unit verts by rotation
    uv = []
    for v in unit_verts:
        uv.append(np.array([ca * v[0] - sa * v[1], sa * v[0] + ca * v[1]]))
    # support of unit hexagon in direction n
    def h_unit(n):
        return max(float(np.dot(n, u)) for u in uv)
    s = 0.0
    for p in pts:
        for n in normals:
            hn = h_unit(n)
            if hn <= 1e-12:
                continue
            val = (p[0] * n[0] + p[1] * n[1]) / hn
            if val > s:
                s = val
    return s


def _objective(params):
    R, ring_rot, inner_rot, outer_rot = params
    # place 12 hexagons on a ring of radius R
    pts = []
    for k in range(N):
        th = ring_rot + math.radians(360.0 * k / N)
        x = R * math.cos(th)
        y = R * math.sin(th)
        pts.append((x, y, inner_rot))
    # check pairwise overlaps
    verts = [_hex_verts(x, y, 1.0, ang) for (x, y, ang) in pts]
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_overlap(verts[i], verts[j]):
                # penalty: measure overlap depth via max penetration on axes
                pen = 0.0
                A, B = verts[i], verts[j]
                for P in (A, B):
                    for idx in range(6):
                        ax = P[idx + 1][0] - P[idx][0]
                        ay = P[idx + 1][1] - P[idx][1]
                        nlen = math.hypot(ax, ay)
                        if nlen < 1e-12:
                            continue
                        nx, ny = -ay / nlen, ax / nlen
                        minA = min(p[0] * nx + p[1] * ny for p in A)
                        maxA = max(p[0] * nx + p[1] * ny for p in A)
                        minB = min(p[0] * nx + p[1] * ny for p in B)
                        maxB = max(p[0] * nx + p[1] * ny for p in B)
                        gap = min(maxA - minB, maxB - minA)
                        if gap < 0:
                            pen = max(pen, -gap)
                return 1e4 + pen * 1000.0
    # outer side length
    side = _outer_side_from_points([(x, y) for (x, y, _) in pts], outer_rot)
    return side


def construct():
    t0 = time.time()
    best = None
    best_val = float('inf')

    # initial guesses: ring radius ~ 2.3 (hexagons on ring, need R > ~2*1.0), rotations
    starts = [
        (2.4, 0.0, 0.0, 0.0),
        (2.6, 0.0, 30.0, 30.0),
        (2.8, 0.0, 0.0, 0.0),
        (2.5, 15.0, 15.0, 15.0),
        (3.0, 0.0, 0.0, 30.0),
    ]
    for s in starts:
        if time.time() - t0 > DEADLINE - 10.0:
            break
        try:
            res = minimize(_objective, s, method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': 1e-4, 'fatol': 1e-4})
        except Exception:
            continue
        if res.fun < best_val:
            best_val = float(res.fun)
            best = res.x
        if time.time() - t0 > DEADLINE - 10.0:
            break

    if best is None:
        # fallback: a valid ring that is guaranteed to fit
        R = 3.5
        pts = []
        for k in range(N):
            th = math.radians(360.0 * k / N)
            pts.append((R * math.cos(th), R * math.sin(th), 0.0))
        side = _outer_side_from_points([(x, y) for (x, y, _) in pts], 0.0)
        return pts, (0.0, 0.0), side, 0.0

    R, ring_rot, inner_rot, outer_rot = best
    pts = []
    for k in range(N):
        th = ring_rot + math.radians(360.0 * k / N)
        x = R * math.cos(th)
        y = R * math.sin(th)
        pts.append((x, y, inner_rot))
    side = _outer_side_from_points([(x, y) for (x, y, _) in pts], outer_rot)
    return pts, (0.0, 0.0), side, outer_rot
