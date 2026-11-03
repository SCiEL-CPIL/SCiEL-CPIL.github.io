"""Packing 12 unit hexagons in a regular hexagon via direct non-convex optimization."""
import math
import time
import numpy as np

N = 12
DEADLINE = 140.0  # safety margin under the 150 s kill
EPS = 1e-6

# Unit hexagon (circumradius 1) vertices for rotation 0
_BASE_VERTS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])


def _rot(pts, c, s):
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T


def _min_outer_side(pos, ang):
    """Minimal side length of an outer regular hexagon (any center/rotation)
    containing the unit hexagons at pos (n,2) with angles ang (n,)."""
    verts = []
    for k in range(N):
        th = ang[k]
        verts.append(_rot(_BASE_VERTS, math.cos(th), math.sin(th)) + pos[k])
    A = np.vstack(verts)  # (6n,2)
    # support function of the set; outer hex side S with center c, rotation phi:
    # S = max_i (h_A(u_i) - c.u_i) + max_i(-h_A(u_i) + c.u_i),  u_i = 6 directions
    best = (None, None, None)
    for phi in np.arange(0.0, math.pi / 3.0, math.pi / 60.0):
        c0, s0 = math.cos(phi), math.sin(phi)
        U = np.array([[math.cos(phi + math.radians(a)), math.sin(phi + math.radians(a))] for a in range(0, 360, 60)])
        hA = A @ U.T  # (6n,6)
        hAmax = hA.max(axis=0)
        hAmin = hA.min(axis=0)
        # minimize over c: S(c) = max_i (hAmax_i - c.u_i) + max_i(-hAmin_i + c.u_i)
        # = max_i (hAmax_i - c.u_i) + max_i ( -hAmin_i + c.u_i )
        # The two terms are convex in c; minimize sum of two convex (each a max of linear).
        # Use small grid over c for speed, then refine.
        # bounding box
        cx = (A[:, 0].min() + A[:, 0].max()) / 2.0
        cy = (A[:, 1].min() + A[:, 1].max()) / 2.0
        R = max(A[:, 0].max() - A[:, 0].min(), A[:, 1].max() - A[:, 1].min()) / 2.0 + 1.5
        found = None
        for gcx in np.linspace(cx - R, cx + R, 7):
            for gcy in np.linspace(cy - R, cy + R, 7):
                c = np.array([gcx, gcy])
                t1 = (hAmax - c @ U.T).max()
                t2 = ((-hAmin) + c @ U.T).max()
                val = t1 + t2
                if found is None or val < found[0]:
                    found = (val, gcx, gcy)
        # refine around best grid point
        val, gx, gy = found
        for _ in range(3):
            R2 = max(0.3, R / 3.0)
            for gcx in np.linspace(gx - R2, gx + R2, 5):
                for gcy in np.linspace(gy - R2, gy + R2, 5):
                    c = np.array([gcx, gcy])
                    t1 = (hAmax - c @ U.T).max()
                    t2 = ((-hAmin) + c @ U.T).max()
                    v = t1 + t2
                    if v < val:
                        val, gx, gy = v, gcx, gcy
        if best[0] is None or val < best[0] - 1e-12:
            best = (val, gx, gy, phi)
    return best  # (side, cx, cy, phi)


def _separating_ok(p1, a1, p2, a2):
    """True if two unit hexagons (center p, angle a) do NOT overlap (touching allowed)."""
    d = p2 - p1
    # check 6 axes from hex1 and 6 from hex2
    for a in (a1, a2):
        c, s = math.cos(a), math.sin(a)
        U = np.array([[math.cos(a + math.radians(k)), math.sin(a + math.radians(k))] for k in range(0, 360, 60)])
        # project both hexagons onto each axis
        # hex1 verts = R1 * base + p1 ; hex2 = R2*base + p2
        # projection of R*base onto u = (R base) . u
        R1 = _rot(_BASE_VERTS, math.cos(a1), math.sin(a1))
        R2 = _rot(_BASE_VERTS, math.cos(a2), math.sin(a2))
        for u in U:
            proj1 = R1 @ u
            proj2 = R2 @ u
            # intervals
            lo1, hi1 = proj1.min(), proj1.max()
            lo2, hi2 = proj2.min(), proj2.max()
            # shift by centers
            lo1 += p1 @ u; hi1 += p1 @ u
            lo2 += p2 @ u; hi2 += p2 @ u
            if hi1 < lo2 - EPS or hi2 < lo1 - EPS:
                return True  # separated on this axis
    return False  # overlapping


def _overlap_penalty(pos, ang):
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if not _separating_ok(pos[i], ang[i], pos[j], ang[j]):
                # estimate penetration by center distance vs min
                dist = np.linalg.norm(pos[i] - pos[j])
                pen += max(0.0, (2.0 - dist)) ** 2
    return pen


def construct():
    t0 = time.time()
    best = None  # (side, pos, ang, cx, cy, phi)

    def try_place(pos, ang):
        nonlocal best
        if time.time() - t0 > DEADLINE:
            return
        # check feasibility
        for i in range(N):
            for j in range(i + 1, N):
                if not _separating_ok(pos[i], ang[i], pos[j], ang[j]):
                    return
        res = _min_outer_side(pos, ang)
        side, cx, cy, phi = res
        if best is None or side < best[0]:
            best = (side, pos.copy(), ang.copy(), cx, cy, phi)

    # ---- Initial good seed: 3x4 grid with small offsets ----
    def seed_grid():
        pos = []
        ang = []
        xs = [-3.0, -1.0, 1.0, 3.0]
        ys = [-1.5, 1.5]
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                pos.append([x, y])
                ang.append(0.0)
        return np.array(pos), np.array(ang)

    # ---- Optimization via SLSQP ----
    def optimize(pos0, ang0):
        nonlocal best
        if time.time() - t0 > DEADLINE:
            return
        from scipy.optimize import minimize

        def x2pos(x):
            return x[:2 * N].reshape(N, 2)

        def pos2x(pos, ang):
            return np.concatenate([pos.reshape(-1), ang])

        def obj(x):
            pos, ang = x2pos(x), x[2 * N:]
            # objective = outer side + penalty for overlap
            pen = _overlap_penalty(pos, ang)
            if pen > 1e-4:
                return pen * 10.0 + 100.0
            res = _min_outer_side(pos, ang)
            return res[0]

        # bounds
        bounds = []
        for _ in range(N):
            bounds += [(-15, 15), (-15, 15)]
        bounds += [(-math.pi / 6, math.pi / 6)] * N

        try:
            res = minimize(obj, pos2x(pos0, ang0), method='SLSQP',
                           bounds=bounds, options={'maxiter': 60, 'ftol': 1e-6})
            if time.time() - t0 > DEADLINE:
                return
            pos, ang = x2pos(res.x), res.x[2 * N:]
            try_place(pos, ang)
        except Exception:
            pass

    # initial
    pos0, ang0 = seed_grid()
    try_place(pos0, ang0)
    optimize(pos0, ang0)

    # random restarts
    rng = np.random.default_rng(0)
    attempts = 0
    while time.time() - t0 < DEADLINE:
        attempts += 1
        # random perturbation of the grid or random placement
        if attempts % 3 == 0:
            # fully random
            pos = rng.uniform(-6, 6, (N, 2))
            ang = rng.uniform(-math.pi / 6, math.pi / 6, N)
        else:
            pos, ang = seed_grid()
            pos = pos + rng.normal(0, 0.4, (N, 2))
            ang = ang + rng.normal(0, 0.15, N)
        optimize(pos, ang)
        if best is None or best[0] < 7.3:
            # good enough, stop early to save time
            if best[0] < 7.0:
                break

    if best is None:
        # fallback: return the seed grid with a safe outer side
        pos, ang = seed_grid()
        side, cx, cy, phi = _min_outer_side(pos, ang)
        best = (side, pos, ang, cx, cy, phi)

    side, pos, ang, cx, cy, phi = best
    inner = [[float(pos[i, 0]), float(pos[i, 1]), float(math.degrees(ang[i]))] for i in range(N)]
    return inner, (float(cx), float(cy)), float(side), float(math.degrees(phi))
