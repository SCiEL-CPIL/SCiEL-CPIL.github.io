import math
import time
import numpy as np
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY = 15.0
MIN_DIST = 2.0 * APOTHEM
MIN_DIST2 = MIN_DIST * MIN_DIST


def _time_ok():
    return time.time() - START_TIME < TIME_LIMIT - SAFETY


def _axial(q, r):
    return np.array([math.sqrt(3) * (q + r / 2.0), 1.5 * r])


def _lp_fit(pts, beta):
    """Exact minimal enclosing hexagon for a fixed rotation beta.
    Returns (S, (cx, cy)) or (inf, None)."""
    a = beta + math.pi / 6.0
    normals = np.column_stack([np.cos(a + np.arange(6) * math.pi / 3.0),
                               np.sin(a + np.arange(6) * math.pi / 3.0)])
    n = len(pts)
    A = np.zeros((12 * n, 3))
    b = np.zeros(12 * n)
    for k in range(6):
        nx, ny = normals[k]
        p = pts @ normals[k]
        s = 2 * k * n
        A[s:s + n] = np.column_stack([-nx * np.ones(n), -ny * np.ones(n), -np.ones(n)])
        b[s:s + n] = -p
        A[s + n:s + 2 * n] = np.column_stack([nx * np.ones(n), ny * np.ones(n), -np.ones(n)])
        b[s + n:s + 2 * n] = p
    try:
        res = linprog(np.array([0.0, 0.0, 1.0]), A_ub=A, b_ub=b,
                      bounds=[(None, None)] * 2 + [(0, None)], method='highs')
        if res.status == 0:
            D = res.x[2]
            return (D + APOTHEM) / APOTHEM, (float(res.x[0]), float(res.x[1]))
    except Exception:
        pass
    return float('inf'), None


def _exact(pts, cx, cy, beta):
    a = beta + math.pi / 6.0
    normals = np.column_stack([np.cos(a + np.arange(6) * math.pi / 3.0),
                               np.sin(a + np.arange(6) * math.pi / 3.0)])
    pc = pts - np.array([cx, cy])
    proj = pc @ normals.T
    D = np.max(np.abs(proj))
    return (D + APOTHEM) / APOTHEM


def _smooth(pts, cx, cy, beta, t):
    a = beta + math.pi / 6.0
    normals = np.column_stack([np.cos(a + np.arange(6) * math.pi / 3.0),
                               np.sin(a + np.arange(6) * math.pi / 3.0)])
    pc = pts - np.array([cx, cy])
    proj = pc @ normals.T
    allp = np.concatenate([proj.ravel(), -proj.ravel()])
    m = np.max(allp)
    if m > 500:
        return float('inf')
    s = (allp - m) / t
    s = np.clip(s, -60, 60)
    D = t * (np.log(np.sum(np.exp(s))) + m / t)
    return (D + APOTHEM) / APOTHEM


def _overlap(pts, tol=1e-7):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            d2 = np.dot(pts[i] - pts[j], pts[i] - pts[j])
            if d2 < MIN_DIST2 - tol:
                return False
    return True


def _force(pts, rng, iters=1400, T0=0.10, T1=0.0008):
    """Lennard-Jones repulsion + mild centering to spread N disjoint hexagons."""
    n = len(pts)
    pts = pts.astype(float).copy()
    for it in range(iters):
        if not _time_ok():
            break
        T = T0 * (T1 / T0) ** (it / max(1, iters - 1))
        F = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                r2 = d @ d
                r = math.sqrt(r2) if r2 > 1e-12 else 1e-6
                r4 = r2 * r2
                # repulsion ~ r^-4 attractive-ish well handled by centering; use pure repulsion
                Fij = (4.0 / r4) * d
                F[i] += Fij
                F[j] -= Fij
        for i in range(n):
            F[i] -= 0.15 * pts[i]  # pull toward origin to keep bounded
        pts += T * F
    return pts


def _local(pts, rng, n_restarts=5, t_final=0.008):
    """Local minimax over (cx, cy, beta) using smooth surrogate -> exact."""
    best_S = float('inf')
    best = None
    cx0 = float(np.mean(pts[:, 0]))
    cy0 = float(np.mean(pts[:, 1]))
    for r in range(n_restarts):
        if not _time_ok():
            break
        cx = cx0 + rng.uniform(-1.0, 1.0)
        cy = cy0 + rng.uniform(-1.0, 1.0)
        beta = rng.uniform(0.0, math.pi / 6.0)
        for t in (0.05, 0.02, 0.008):
            if not _time_ok():
                break

            def obj(p, t=t):
                cx, cy, b = p
                b = ((b % (math.pi / 6.0)) + math.pi / 6.0) % (math.pi / 6.0)
                return _smooth(pts, cx, cy, b, t)

            try:
                res = minimize(obj, [cx, cy, beta], method='Nelder-Mead',
                               options={'maxiter': 800, 'xatol': 1e-11,
                                        'fatol': 1e-11, 'adaptive': True})
                cx, cy, beta = res.x
            except Exception:
                pass
        beta = ((beta % (math.pi / 6.0)) + math.pi / 6.0) % (math.pi / 6.0)
        S, c = _lp_fit(pts, beta)
        if c is not None and S < best_S:
            best_S = S
            best = (S, c[0], c[1], beta)
    return best


def construct():
    if not _time_ok():
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    rng = np.random.default_rng(12345)
    best_S = float('inf')
    best_pts = None
    best_beta = 0.0
    best_center = None

    # Seed initial layouts: ring + center (natural hex packing) plus random.
    seeds = []
    # ring of 6 around center
    ring = []
    for k in range(6):
        ang = k * math.pi / 3.0
        ring.append([MIN_DIST * math.cos(ang), MIN_DIST * math.sin(ang)])
    ring.append([0.0, 0.0])
    seeds.append(np.array(ring))
    # 11 random
    for _ in range(10):
        p = rng.uniform(-3.0, 3.0, size=(N, 2))
        seeds.append(p)

    for seed in seeds:
        if not _time_ok():
            break
        # force-directed relaxation (a few different random restarts)
        for rep in range(3):
            if not _time_ok():
                break
            pert = seed + rng.uniform(-0.3, 0.3, size=(N, 2))
            pts = _force(pert, rng)
            if not _overlap(pts):
                continue
            res = _local(pts, rng)
            if res is not None:
                S, cx, cy, beta = res
                if S < best_S:
                    best_S = S
                    best_pts = pts.copy()
                    best_center = (cx, cy)
                    best_beta = beta

    if best_pts is None:
        # fallback: simple axial arrangement
        axial = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1),
                 (1, -1), (1, 1), (-1, 2), (-2, 1), (2, -1)]
        best_pts = np.array([_axial(q, r) for q, r in axial])
        best_beta = 0.0
        S, c = _lp_fit(best_pts, best_beta)
        best_S = S
        best_center = c if c is not None else (0.0, 0.0)

    cx, cy = best_center

    # Final global polish: refine (cx, cy, beta) once more with exact objective.
    def exact_obj(p):
        cx, cy, b = p
        b = ((b % (math.pi / 6.0)) + math.pi / 6.0) % (math.pi / 6.0)
        return _exact(best_pts, cx, cy, b)

    for _ in range(8):
        if not _time_ok():
            break
        init = [cx, cy, best_beta + rng.uniform(-0.02, 0.02)]
        try:
            res = minimize(exact_obj, init, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-13,
                                    'fatol': 1e-13, 'adaptive': True})
            if res.fun < best_S:
                cx, cy, b = res.x
                b = ((b % (math.pi / 6.0)) + math.pi / 6.0) % (math.pi / 6.0)
                best_S, c = _lp_fit(best_pts, b)
                if c is not None:
                    best_center = c
                    best_beta = b
        except Exception:
            pass

    cx, cy = best_center
    angle_deg = math.degrees(best_beta)

    inner = []
    for x, y in best_pts:
        inner.append([float(x), float(y), float(angle_deg)])
    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(angle_deg)
    return inner, center, side, rotation
