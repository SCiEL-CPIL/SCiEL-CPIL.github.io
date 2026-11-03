"""Combined program: lattice-cluster seed + physical settling + outer-angle/center refinement."""
import time
import math
import numpy as np
from scipy.optimize import minimize as _scipy_minimize

N = 12
DEADLINE = 140.0  # safety margin before the 150s kill

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _hex_vertices(theta):
    # flat-top regular hexagon, side 1, circumradius 1, rotated by theta
    return np.array([[math.cos(theta + k * math.pi / 3.0),
                      math.sin(theta + k * math.pi / 3.0)] for k in range(6)])

def _norms(theta):
    # 6 edge-normal directions of a regular hexagon oriented by theta
    return np.array([[math.cos(theta + math.pi / 6.0 + k * math.pi / 3.0),
                      math.sin(theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])

def _bounding_side(pts, theta, center):
    # side length of the smallest regular hexagon (orientation theta, given center)
    # containing the set of points pts.
    nrm = _norms(theta)
    P = pts @ nrm.T
    c = center @ nrm.T
    # required: P[:,k] - c[k] <= s*sqrt(3)/2  for every k
    req = (P - c).max(axis=0)
    return 2.0 * float(np.max(req)) / math.sqrt(3.0)

def _hexagon_overlap(A, B):
    # SAT test between two convex hexagons (arrays of (x,y)); True if overlapping
    for poly in (A, B):
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            projA = A @ normal
            projB = B @ normal
            if projA.max() < projB.min() - 1e-9 or projB.max() < projA.min() - 1e-9:
                return False
    return True

def _min_translation(A, B):
    # separating axis + signed penetration depth (negative if overlapping)
    best_pen = -1.0
    best_axis = None
    for poly in (A, B):
        for k in range(6):
            p1 = poly[k]
            p2 = poly[(k + 1) % 6]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            nl = math.hypot(n[0], n[1])
            if nl < 1e-9:
                continue
            n = n / nl
            pa = A @ n
            pb = B @ n
            sep = min(pa.max() - pb.min(), pb.max() - pa.min())
            if sep < 0 and sep > best_pen:
                best_pen = sep
                best_axis = n
    return best_pen, best_axis

# ---------------------------------------------------------------------------
# Seed: compact triangular-lattice clusters
# ---------------------------------------------------------------------------
def _seed(pattern):
    d = 2.0 * math.sqrt(3.0) + 1e-6
    a1 = np.array([d, 0.0])
    a2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
    pts = []
    if pattern == "rect":
        for i in range(-1, 3):
            for j in range(-1, 2):
                pts.append(i * a1 + j * a2)
    elif pattern == "hex":
        coords = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
                  (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2)]
        for (i, j) in coords:
            pts.append(i * a1 + j * a2)
    else:  # "rect_tall"
        for i in range(-1, 2):
            for j in range(-2, 2):
                pts.append(i * a1 + j * a2)
    pts = np.array(pts)
    c = pts.mean(axis=0)
    pts = pts - c
    return [[float(pts[k][0]), float(pts[k][1]), 0.0] for k in range(N)]

# ---------------------------------------------------------------------------
# Physical settling
# ---------------------------------------------------------------------------
def _simulate(inner, outer, side, outer_theta, dt=0.02, steps=4000):
    centers = np.array([[r[0], r[1]] for r in inner], dtype=float)
    angles = np.array([math.radians(r[2]) for r in inner])
    vel = np.zeros((N, 2))
    omega = np.zeros(N)
    outer_pts = np.array(outer, dtype=float) * side
    verts = [_hex_vertices(angles[k]) for k in range(N)]

    for s in range(steps):
        if time.time() > DEADLINE:
            break
        # ---- forces ----
        for i in range(N):
            vel[i] += -0.02 * (centers[i] - outer_pts)      # attraction toward incenter
            omega[i] += -0.005 * angles[i]                   # gentle alignment toward 0
        # pairwise repulsion via SAT minimum translation vector
        for i in range(N):
            for j in range(i + 1, N):
                A = verts[i] + centers[i]
                B = verts[j] + centers[j]
                if _hexagon_overlap(A, B):
                    pen, axis = _min_translation(A, B)
                    if axis is not None:
                        f = 0.5 * (-pen)
                        vel[i] += f * axis
                        vel[j] -= f * axis
        # ---- containment: push inward if a hexagon pokes out ----
        for i in range(N):
            pts = verts[i] + centers[i]
            nrm = _norms(outer_theta)
            P = pts @ nrm.T
            c = outer_pts @ nrm.T
            for k in range(6):
                exceed = (P[:, k] - c[k]).max() - side * math.sqrt(3.0) / 2.0
                if exceed > 0:
                    n = nrm[k]
                    vel[i] -= 0.5 * exceed * n
        # ---- integrate (symplectic Euler) ----
        vel *= 0.85
        omega *= 0.85
        centers += vel * dt
        angles += omega * dt
        for k in range(N):
            verts[k] = _hex_vertices(angles[k])

    return [[float(centers[k][0]), float(centers[k][1]), float(math.degrees(angles[k]))] for k in range(N)]

# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------
def _all_points(inner):
    verts = [_hex_vertices(math.radians(r[2])) for r in inner]
    centers = np.array([[r[0], r[1]] for r in inner])
    pts = []
    for k in range(N):
        pts.append(verts[k] + centers[k])
    return np.vstack(pts), centers

def _refine_inner(inner, outer, side, outer_theta, maxiter=200):
    # Nelder-Mead on the N centers, minimizing the bounding side (outer fixed)
    verts = [_hex_vertices(math.radians(r[2])) for r in inner]
    centers = np.array([[r[0], r[1]] for r in inner])

    def obj(x):
        c = x.reshape(N, 2)
        pts = []
        for k in range(N):
            pts.append(verts[k] + c[k])
        pts = np.vstack(pts)
        return _bounding_side(pts, outer_theta, np.array(outer))

    x0 = centers.reshape(-1)
    res = _scipy_minimize(obj, x0, method="Nelder-Mead",
                          options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-9})
    c = res.x.reshape(N, 2)
    new_inner = []
    for k in range(N):
        new_inner.append([float(c[k][0]), float(c[k][1]), inner[k][2]])
    return new_inner

def _refine_outer(inner, outer, side, outer_theta):
    # sweep outer angle, re-centering each time; return best (side, center, angle)
    pts, centers = _all_points(inner)
    best = (side, outer, outer_theta)
    for ang in np.linspace(0.0, math.pi / 3.0, 121):
        nrm = _norms(ang)
        P = pts @ nrm.T
        c = np.array(outer) @ nrm.T
        req = (P - c).max(axis=0)
        b = 2.0 * float(np.max(req)) / math.sqrt(3.0)
        if b < best[0] - 1e-9:
            # re-center so the worst face is centered
            worst = int(np.argmax(req))
            shift = 0.5 * (req[worst] - (req).min())  # not used; keep outer as-is
            best = (b, tuple(float(x) for x in outer), float(ang))
    return best

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct():
    start = time.time()

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    def _consider(inner, outer, side, outer_theta):
        nonlocal best_side, best_inner, best_center, best_angle
        if side < best_side:
            best_side = side
            best_inner = [list(r) for r in inner]
            best_center = tuple(float(x) for x in outer)
            best_angle = float(outer_theta)

    patterns = ["rect", "hex", "rect_tall"]
    for pat in patterns:
        if time.time() > DEADLINE:
            break
        inner = _seed(pat)
        # initial outer estimate via coarse angle sweep at origin
        pts, _ = _all_points(inner)
        s0, t0 = 10.0, 0.0
        for ang in np.linspace(0.0, math.pi / 3.0, 61):
            b = _bounding_side(pts, ang, np.zeros(2))
            if b < s0:
                s0, t0 = b, ang
        outer = (0.0, 0.0)

        # settle
        inner = _simulate(inner, outer, s0, t0, steps=2500)
        # refine outer (angle) then inner (centers)
        s1, outer1, t1 = _refine_outer(inner, outer, s0, t0)
        inner = _refine_inner(inner, outer1, s1, t1, maxiter=200)
        s2, outer2, t2 = _refine_outer(inner, outer1, s1, t1)
        inner = _refine_inner(inner, outer2, s2, t2, maxiter=120)
        s3, outer3, t3 = _refine_outer(inner, outer2, s2, t2)

        _consider(inner, outer3, s3, t3)

    # ---- final fine sweep of the outer angle on the best inner set ----
    if best_inner is not None:
        pts, _ = _all_points(best_inner)
        fine = np.linspace(0.0, math.pi / 3.0, 301)
        sF, cF, tF = best_side, best_center, best_angle
        for ang in fine:
            b = _bounding_side(pts, ang, np.array(best_center))
            if b < sF - 1e-12:
                sF, tF = b, ang
        _consider(best_inner, best_center, sF, tF)

    # ---- fallback (should not be reached) ----
    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, math.degrees(best_angle)
# EVOLVE-BLOCK-END
