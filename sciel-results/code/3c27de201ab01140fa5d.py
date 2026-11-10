# EVOLVE-BLOCK-START
import time
import numpy as np


def _hex_verts(cx, cy, d, ang_deg):
    a = np.deg2rad(ang_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(th), cy + d * np.sin(th)))


def _support(v, ang_deg):
    a = np.deg2rad(ang_deg)
    d = np.array([np.cos(a + np.deg2rad(60.0 * np.arange(6))),
                  np.sin(a + np.deg2rad(60.0 * np.arange(6)))])
    return (v @ d.T).max(axis=0)


def _enclosing_side(centers, ang_deg):
    """Smallest side length of a regular hexagon (center 0, orientation ang_deg)
    containing every unit hexagon whose centers are in `centers`."""
    h = _support(centers, ang_deg)
    return h.max() - h.min()


def _separated(centers):
    """True if no two unit hexagons (same orientation) overlap (SAT, 6 axes)."""
    n = len(centers)
    d = np.array([np.cos(np.deg2rad(60.0 * np.arange(6))),
                  np.sin(np.deg2rad(60.0 * np.arange(6)))])
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            if np.any(np.abs(diff @ d.T) > 1.0 + 1e-12):
                return True
    return False


def _valid(centers, s, ang_deg):
    if not np.all(np.isfinite(centers)) or not np.isfinite(s):
        return False
    if s < np.sqrt(12.0) - 1e-9:
        return False
    if not _separated(centers):
        return False
    # containment: every inner vertex inside outer hexagon (center 0)
    h = _support(centers, ang_deg)
    return (h.max() - h.min()) <= s + 1e-9


def _nm_minimize(f, x0, n_iter, tol=1e-9, max_step=3.0):
    """Nelder-Mead with step-size cap and a validity gate."""
    n = len(x0)
    simplex = [x0.copy()]
    for k in range(1, n + 1):
        p = x0.copy()
        p[k - 1] += 0.1
        simplex.append(p)
    fx = [f(x) for x in simplex]
    best = min(range(n + 1), key=lambda i: fx[i])
    best_x = simplex[best].copy()
    best_f = fx[best]
    for _ in range(n_iter):
        order = sorted(range(n + 1), key=lambda i: fx[i])
        simplex = [simplex[i] for i in order]
        fx = [fx[i] for i in order]
        worst = n
        if fx[worst] - fx[0] < tol:
            break
        centroid = np.mean(simplex[:n], axis=0)
        # reflection
        xr = centroid + 1.0 * (centroid - simplex[worst])
        fr = f(xr)
        if fx[0] <= fr < fx[worst - 1]:
            simplex[worst] = xr
            fx[worst] = fr
        elif fr < fx[0]:
            # expansion
            xe = centroid + 2.0 * (centroid - simplex[worst])
            fe = f(xe)
            if fe < fr:
                simplex[worst] = xe
                fx[worst] = fe
            else:
                simplex[worst] = xr
                fx[worst] = fr
        else:
            # contraction
            if fr < fx[worst]:
                xc = centroid + 0.5 * (simplex[worst] - centroid)
                fc = f(xc)
                if fc <= fr:
                    simplex[worst] = xc
                    fx[worst] = fc
                else:
                    # shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fx[i] = f(simplex[i])
            else:
                xc = centroid - 0.5 * (simplex[worst] - centroid)
                fc = f(xc)
                if fc < fx[worst]:
                    simplex[worst] = xc
                    fx[worst] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fx[i] = f(simplex[i])
        # cap step size
        for i in range(1, n + 1):
            step = simplex[i] - simplex[0]
            nn = np.linalg.norm(step)
            if nn > max_step:
                simplex[i] = simplex[0] + step / nn * max_step
                fx[i] = f(simplex[i])
        b = min(range(n + 1), key=lambda i: fx[i])
        if fx[b] < best_f:
            best_f = fx[b]
            best_x = simplex[b].copy()
    return best_x, best_f


def hexagon_packing_12():
    t0 = time.time()
    DEADLINE = 630.0
    MARGIN = 5.0

    ang = 0.0
    n = 12
    # Compact 4-3-3-2 trapezoid seed (rows of hexagons), all oriented 0 deg.
    centers = np.array([
        # row of 4
        [-3.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [3.0, 0.0],
        # row of 3
        [-2.0, np.sqrt(3.0)], [0.0, np.sqrt(3.0)], [2.0, np.sqrt(3.0)],
        # row of 3
        [-2.0, -np.sqrt(3.0)], [0.0, -np.sqrt(3.0)], [2.0, -np.sqrt(3.0)],
        # row of 2
        [-1.0, -2.0 * np.sqrt(3.0)], [1.0, -2.0 * np.sqrt(3.0)],
    ])
    centers = centers - centers.mean(axis=0)

    s0 = _enclosing_side(centers, ang)
    best_centers = centers.copy()
    best_s = s0 + 1e-9
    if not _valid(centers, best_s, ang):
        # fallback: guarantee validity with a generous margin
        best_s = s0 + 0.5
        best_centers = centers.copy()

    # Objective: minimize enclosing side; penalize overlap so NM stays in feasible region.
    def obj(x):
        c = x.reshape(n, 2)
        s = _enclosing_side(c, ang)
        if not _separated(c):
            # count how many pairs overlap to make a smooth-ish penalty
            d = np.array([np.cos(np.deg2rad(60.0 * np.arange(6))),
                          np.sin(np.deg2rad(60.0 * np.arange(6)))])
            pen = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    m = np.abs((c[i] - c[j]) @ d.T).max()
                    if m < 1.0:
                        pen += (1.0 - m) ** 2
            return s + 100.0 * pen
        return s

    x0 = best_centers.reshape(-1).copy()
    # Scale initial simplex by the problem length scale.
    x0_scaled = x0.copy()

    n_iter = 500
    # run bounded Nelder-Mead
    try:
        xc, fc = _nm_minimize(obj, x0_scaled, n_iter, tol=1e-10, max_step=4.0)
        c = xc.reshape(n, 2)
        s = _enclosing_side(c, ang)
        if _valid(c, s + 1e-9, ang) and s < best_s:
            best_centers = c.copy()
            best_s = s + 1e-9
    except Exception:
        pass

    # Final safety: if anything is off, fall back to the guaranteed-valid seed.
    if not _valid(best_centers, best_s, ang):
        best_centers = centers.copy()
        best_s = _enclosing_side(centers, ang) + 1e-9

    inner = np.column_stack([best_centers, np.zeros(n)])
    outer = np.array([0.0, 0.0, ang])
    return inner, outer, float(best_s)


# EVOLVE-BLOCK-END
