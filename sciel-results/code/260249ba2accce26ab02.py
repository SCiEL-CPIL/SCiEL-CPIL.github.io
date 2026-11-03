"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
TOL = 1e-12
DEADLINE = 120.0

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon (flat-top at angle 0) centered at (cx,cy)."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)


def _rel_dist_fast(alpha):
    """Min center distance for two unit hexagons with relative rotation alpha (deg) not to overlap."""
    vx = [math.cos(i * math.pi / 3.0) for i in range(6)]
    vy = [math.sin(i * math.pi / 3.0) for i in range(6)]
    nx = [math.cos(i * math.pi / 3.0 + math.pi / 6.0) for i in range(6)]
    ny = [math.sin(i * math.pi / 3.0 + math.pi / 6.0) for i in range(6)]
    best = 0.0
    for i in range(6):
        ex = vx[i] - 0.5 * vx[i + 1]
        ey = vy[i] - 0.5 * vy[i + 1]
        for j in range(6):
            d = nx[j] * ex + ny[j] * ey
            if d > best:
                best = d
            d = -nx[j] * ex - ny[j] * ey
            if d > best:
                best = d
    best += 1.0
    return best  # scalar, invariant under rotation


def objective(v):
    """Outer side length needed for config v, plus soft overlap penalties."""
    cx, cy = v[0], v[1]
    theta = math.radians(v[2])
    ct, st = math.cos(theta), math.sin(theta)
    ux, uy = ct, st
    vx, vy = -st, ct
    n1 = (math.sqrt(3.0) / 2.0) * ux - 0.5 * uy
    n2 = (math.sqrt(3.0) / 2.0) * ux + 0.5 * uy
    n3 = (math.sqrt(3.0) / 2.0) * vx - 0.5 * vy
    n4 = (math.sqrt(3.0) / 2.0) * vx + 0.5 * vy
    n5 = (math.sqrt(3.0) / 2.0) * (-ux) - 0.5 * (-uy)
    n6 = (math.sqrt(3.0) / 2.0) * (-ux) + 0.5 * (-uy)

    px = np.zeros(N)
    py = np.zeros(N)
    ang = np.zeros(N)
    for k in range(N):
        b = 3 + 3 * k
        px[k] = v[b]
        py[k] = v[b + 1]
        ang[k] = v[b + 2]

    m1 = m2 = m3 = -1e18
    for k in range(N):
        a = math.radians(ang[k])
        for i in range(6):
            t = a + i * math.pi / 3.0
            X = px[k] + math.cos(t)
            Y = py[k] + math.sin(t)
            p1 = n1 * X + n2 * Y
            if p1 > m1:
                m1 = p1
            p2 = n3 * X + n4 * Y
            if p2 > m2:
                m2 = p2
            p3 = n5 * X + n6 * Y
            if p3 > m3:
                m3 = p3
    side = max(m1 - (n1 * cx + n2 * cy),
               m2 - (n3 * cx + n4 * cy),
               m3 - (n5 * cx + n6 * cy))

    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = px[i] - px[j]
            dy = py[i] - py[j]
            d = math.hypot(dx, dy)
            rel = abs(ang[i] - ang[j]) % 60.0
            if rel > 30.0:
                rel = 60.0 - rel
            need = _rel_dist_fast(rel)
            if d < need:
                pen += (need - d) ** 2
    return side + 1000.0 * pen


def solve_from(x0, deadline):
    res = minimize(objective, x0, method='L-BFGS-B',
                   options={'maxiter': 400, 'maxfun': 100000, 'ftol': 1e-12, 'gtol': 1e-9})
    best, bestf = res.x, res.fun
    if time.time() < deadline:
        res2 = minimize(objective, best, method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res2.fun < bestf:
            best, bestf = res2.x, res2.fun
    return best, bestf


def _clean(v):
    cx, cy = v[0], v[1]
    inner = []
    for k in range(N):
        b = 3 + 3 * k
        inner.append([v[b], v[b + 1], v[b + 2]])
    side = objective(v)
    return inner, (cx, cy), side, v[2]


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - 10.0
    rng = np.random.default_rng(12345)

    best_v = None
    best_f = 1e18

    def consider(v, f):
        nonlocal best_v, best_f
        if f < best_f:
            best_v, best_f = v, f

    # Phase 1: structured hex-lattice initial configs
    s_vals = [2.0, 2.05, 2.1, 2.15, 2.2]
    for s in s_vals:
        if time.time() > deadline:
            break
        ax = [(0, 0)]
        for q in range(-2, 3):
            for rr in range(-2, 3):
                if len(ax) >= 12:
                    break
                if (q, rr) in ax:
                    continue
                ax.append((q, rr))
        ax = ax[:12]
        x0 = np.zeros(3 + 3 * N)
        for k, (q, rr) in enumerate(ax):
            b = 3 + 3 * k
            x0[b] = s * 1.5 * q
            x0[b + 1] = s * math.sqrt(3.0) * (rr + q / 2.0)
            x0[b + 2] = 0.0
        v, f = solve_from(x0, deadline)
        consider(v, f)

    # Phase 2: random restarts
    for r in range(60):
        if time.time() > deadline:
            break
        x0 = np.zeros(3 + 3 * N)
        x0[2] = rng.uniform(0, 60)
        for k in range(N):
            b = 3 + 3 * k
            ang = rng.uniform(0, 2 * math.pi)
            rad = rng.uniform(0, 4.0)
            x0[b] = rad * math.cos(ang)
            x0[b + 1] = rad * math.sin(ang)
            x0[b + 2] = rng.uniform(0, 60)
        v, f = solve_from(x0, deadline)
        consider(v, f)

    # Phase 3: perturb best
    if best_v is not None and time.time() < deadline:
        for it in range(8):
            if time.time() > deadline:
                break
            pert = best_v + rng.normal(0, 0.3, size=best_v.shape)
            v, f = solve_from(pert, deadline)
            consider(v, f)

    if best_v is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return _clean(best_v)

# EVOLVE-BLOCK-END
