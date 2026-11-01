"""Pack N unit regular hexagons inside a regular hexagon, minimizing outer side length."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 19  # hexagonal flower: 1 + 6 + 12

_DEADLINE = 140.0
_T0 = time.time()


def _hex_vertices(cx, cy, s, ang):
    """Vertices of a regular hexagon, center (cx,cy), side s, rotation ang (rad)."""
    a = np.arange(6) * (np.pi / 3.0) + ang
    return np.stack([cx + s * np.cos(a), cy + s * np.sin(a)], axis=1)


def _hexagon_side_length(points):
    """Minimum side length s of a regular hexagon (center, side, rot) enclosing all 2D points."""
    P = np.asarray(points, dtype=float)
    if P.size == 0:
        return 0.0
    best = np.inf
    for ox in range(-1, 2):
        for oy in range(-1, 2):
            cx = P[:, 0].mean() + ox * 0.05
            cy = P[:, 1].mean() + oy * 0.05
            for k in range(3):
                ang = k * (np.pi / 3.0)
                # outward normals of the 6 sides
                n = np.cos(np.arange(6) * (np.pi / 3.0) + ang)
                m = np.sin(np.arange(6) * (np.pi / 3.0) + ang)
                d = (P - np.array([cx, cy])) @ np.stack([n, m], axis=1)  # (N,6)
                dmax = d.max(axis=0)
                # support distance on side k must equal apothem = s*sqrt(3)/2
                # side k and opposite side k+3 give the constraint
                k1 = k
                k2 = (k + 3) % 6
                req = (dmax[k1] + dmax[k2]) / (np.sqrt(3.0))
                s = max(req, 0.0)
                if s < best:
                    best = s
    return best


def _build_cluster():
    """Return list of (x, y) for a symmetric hexagonal cluster of N hexagons."""
    q = int(round((N - 1) / 6.0))  # 1 for N=19
    if q < 1:
        q = 1
    pts = [(0.0, 0.0)]
    for r in range(1, q + 1):
        for a in range(6):
            base = np.array([math.cos(a * np.pi / 3.0), math.sin(a * np.pi / 3.0)])
            for t in range(r):
                step = np.array([math.cos((a + 1) * np.pi / 3.0), math.sin((a + 1) * np.pi / 3.0)])
                p = r * base + t * step
                pts.append((float(p[0]), float(p[1])))
    return pts[:N]


def _objective(z, pts, beta):
    # z = [x0,y0,...,xN-1,yN-1, ang]
    n = len(pts)
    pos = z[:2 * n].reshape(n, 2)
    ang = z[2 * n]
    # vertices of all inner hexagons (unit side, common orientation)
    a = np.arange(6) * (np.pi / 3.0) + ang
    ca = np.cos(a)
    sa = np.sin(a)
    verts = pos[:, None, :] + np.stack([ca, sa], axis=1)[None, :, :]  # (n,6,2)
    V = verts.reshape(-1, 2)

    # auxiliary variable t = outer side length (epigraph), soft-min over 6 sides
    cx = V[:, 0].mean()
    cy = V[:, 1].mean()
    nrm = np.stack([np.cos(np.arange(6) * np.pi / 3.0), np.sin(np.arange(6) * np.pi / 3.0)], axis=1)
    d = (V - np.array([cx, cy])) @ nrm.T  # (M,6)
    dmax = d.max(axis=0)  # (6,)
    # side k & k+3 constraint: (dmax[k]+dmax[k+3])/sqrt(3) <= t
    svals = np.zeros(6)
    for k in range(6):
        svals[k] = (dmax[k] + dmax[(k + 3) % 6]) / np.sqrt(3.0)
    # soft-min of the 6 side values (we want the max, but t must cover all; use max via log-sum-exp)
    m = svals.max()
    soft = m + np.log(np.sum(np.exp(svals - m))) / beta
    # epigraph: t >= soft ; penalize violation
    t = z[2 * n + 1]
    pen = 0.0
    if t < soft:
        pen = 1e4 * (soft - t) ** 2
    return t + pen


def _objective_simple(z, pts, beta):
    n = len(pts)
    pos = z[:2 * n].reshape(n, 2)
    ang = z[2 * n]
    a = np.arange(6) * (np.pi / 3.0) + ang
    ca = np.cos(a)
    sa = np.sin(a)
    verts = pos[:, None, :] + np.stack([ca, sa], axis=1)[None, :, :]
    V = verts.reshape(-1, 2)
    # objective = min over (center, rot) of enclosing side length (soft)
    # use centroid center and 3 rotations, soft-min over the 6 sides
    cx = V[:, 0].mean()
    cy = V[:, 1].mean()
    best = np.inf
    for k in range(3):
        ang2 = k * np.pi / 3.0
        nrm = np.stack([np.cos(np.arange(6) * np.pi / 3.0 + ang2),
                        np.sin(np.arange(6) * np.pi / 3.0 + ang2)], axis=1)
        d = (V - np.array([cx, cy])) @ nrm.T
        dmax = d.max(axis=0)
        svals = np.zeros(6)
        for kk in range(6):
            svals[kk] = (dmax[kk] + dmax[(kk + 3) % 6]) / np.sqrt(3.0)
        m = svals.max()
        soft = m + np.log(np.sum(np.exp(svals - m))) / beta
        if soft < best:
            best = soft
    return best


def construct():
    global _T0
    t_start = time.time()
    pts = _build_cluster()
    n = len(pts)

    # initial guess: hexagonal lattice spacing 2, 60-deg aligned
    def to_xy(q_coords):
        # axial -> cartesian for pointy/flat? use basis vectors at 60 deg, spacing 2
        ex = np.array([2.0, 0.0])
        ey = np.array([2.0 * math.cos(math.pi / 3.0), 2.0 * math.sin(math.pi / 3.0)])
        return ex * q_coords[0] + ey * q_coords[1]

    # build q-coordinates for cluster (same as _build_cluster but in lattice coords)
    q = int(round((N - 1) / 6.0))
    qcoords = [(0, 0)]
    for r in range(1, q + 1):
        for a in range(6):
            base = np.array([math.cos(a * np.pi / 3.0), math.sin(a * np.pi / 3.0)])
            for t in range(r):
                step = np.array([math.cos((a + 1) * np.pi / 3.0), math.sin((a + 1) * np.pi / 3.0)])
                p = r * base + t * step
                qcoords.append((float(p[0]), float(p[1])))
    qcoords = qcoords[:n]

    # lattice basis
    ex = np.array([2.0, 0.0])
    ey = np.array([2.0 * math.cos(math.pi / 3.0), 2.0 * math.sin(math.pi / 3.0)])
    init_pos = []
    for (u, v) in qcoords:
        init_pos.append(ex * u + ey * v)
    init_pos = np.array(init_pos)

    z0 = np.zeros(2 * n + 2)
    z0[:2 * n] = init_pos.reshape(-1)
    z0[2 * n] = 0.0          # angle
    z0[2 * n + 1] = 3.0      # t initial

    # objective with auxiliary t and soft-min
    beta = 40.0
    def obj(z):
        return _objective(z, pts, beta)

    res = None
    best_val = np.inf
    try:
        res = minimize(obj, z0, method='L-BFGS-B',
                       options={'maxiter': 400, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
        if res.success or res.fun < best_val:
            best_val = res.fun
    except Exception:
        pass

    if time.time() - t_start > _DEADLINE - 25.0:
        # safety: return canonical cluster with computed side
        s = _hexagon_side_length(init_pos)
        inner = [[float(p[0]), float(p[1]), 0.0] for p in init_pos]
        return inner, (0.0, 0.0), float(s), 0.0

    # polish: high-precision on positions + angle, using simple objective (no t)
    if res is not None and res.fun < best_val:
        z0 = res.x
    beta2 = 200.0
    def obj2(z):
        return _objective_simple(z[:2 * n + 1], pts, beta2)

    zpol = z0[:2 * n + 1].copy()
    try:
        res2 = minimize(obj2, zpol, method='L-BFGS-B',
                        options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
        if res2.fun < best_val:
            zpol = res2.x
            best_val = res2.fun
    except Exception:
        pass

    # final: extract positions + angle, compute exact enclosing side
    final_pos = zpol[:2 * n].reshape(n, 2)
    final_ang = zpol[2 * n]
    # recompute exact side length (hard)
    s = _hexagon_side_length(final_pos)
    # center: use centroid of all vertices for reporting
    a = np.arange(6) * (np.pi / 3.0) + final_ang
    verts = final_pos[:, None, :] + np.stack([np.cos(a), np.sin(a)], axis=1)[None, :, :]
    V = verts.reshape(-1, 2)
    cx = float(V[:, 0].mean())
    cy = float(V[:, 1].mean())

    # If the optimized side is somehow worse than canonical, fall back
    s_canon = _hexagon_side_length(init_pos)
    if s > s_canon + 1e-6:
        final_pos = init_pos
        final_ang = 0.0
        s = s_canon
        cx = 0.0
        cy = 0.0

    inner = [[float(p[0]), float(p[1]), float(math.degrees(final_ang))] for p in final_pos]
    return inner, (cx, cy), float(s), float(math.degrees(final_ang))
