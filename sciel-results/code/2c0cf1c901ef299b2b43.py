"""Pack N unit regular hexagons (translate + rotate) inside a regular hexagon,
minimizing the outer side length.  Uses a smooth convex objective with L-BFGS
over (positions, rotations, outer side, outer rotation), seeded from a tight
hexagonal close-packed (HCP) layout."""
import math
import time
import numpy as np

N = 11

_T0 = time.time()
_DEADLINE = 112.0          # safety margin under the 150 s wall-clock kill
_SQRT3 = math.sqrt(3.0)


def _support(c):
    """Support radius of a unit regular hexagon in direction angle(s) c (radians).

    s(c) = max_k |cos(c - (60*k + 30))|, 60-degree periodic, in [0.5, 1].
    """
    c = np.asarray(c, dtype=float)
    b = c * (6.0 / math.pi) + 0.5        # in units of 60 degrees, +30 offset
    b = b - np.round(b)                  # wrap to [-0.5, 0.5]
    return np.cos(math.pi * b)           # == |cos(c - (60k+30))|


def _init_hcp():
    """Tight hexagonal close-packed starting layout (flat-top hexagons, spacing sqrt(3))."""
    d = _SQRT3
    rows = [
        [(-1.5 * d, -1.5 * d), (-0.5 * d, -1.5 * d), (0.5 * d, -1.5 * d), (1.5 * d, -1.5 * d)],
        [(-1.0 * d, -0.5 * d), (0.0 * d, -0.5 * d), (1.0 * d, -0.5 * d)],
        [(-1.5 * d, 0.5 * d), (-0.5 * d, 0.5 * d), (0.5 * d, 0.5 * d), (1.5 * d, 0.5 * d)],
    ]
    pts = [p for r in rows for p in r]
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    th = np.zeros(N)
    cx = np.array([0.0])
    cy = np.array([0.0])
    s = np.array([3.6])
    phi = np.array([0.0])
    return np.concatenate([x, y, th, cx, cy, s, phi])


def _make_eval():
    base = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    dphi = np.array([0.0, math.pi / 3.0, 2.0 * math.pi / 3.0,
                     math.pi, 4.0 * math.pi / 3.0, 5.0 * math.pi / 3.0])
    n = N

    def evalf(z):
        x = z[0:n]
        y = z[n:2 * n]
        th = z[2 * n:3 * n]
        cx = z[3 * n]
        cy = z[3 * n + 1]
        s = z[3 * n + 2]
        phi = z[3 * n + 3]
        dx = x - cx
        dy = y - cy
        c = np.cos(phi)
        sn = np.sin(phi)
        # projected distance from each hexagon center to each of the 6 sides
        d = np.empty((n, 6))
        d[:, 0] = c * dx + sn * dy
        d[:, 1] = sn * dx - c * dy
        d[:, 2] = -c * dx + sn * dy
        d[:, 3] = -sn * dx - c * dy
        d[:, 4] = c * dx - sn * dy
        d[:, 5] = -sn * dx + c * dy
        # support radius of the unit hexagon in each side-normal direction
        sup = _support(th[None, :] + base[None, :] + dphi[None, :])
        # constraint: distance to each side >= support radius  ->  s >= max(...)
        return np.max(s - d + sup)

    def evalf_jac(z):
        x = z[0:n]
        y = z[n:2 * n]
        th = z[2 * n:3 * n]
        cx = z[3 * n]
        cy = z[3 * n + 1]
        s = z[3 * n + 2]
        phi = z[3 * n + 3]
        dx = x - cx
        dy = y - cy
        c = np.cos(phi)
        sn = np.sin(phi)
        d = np.empty((n, 6))
        d[:, 0] = c * dx + sn * dy
        d[:, 1] = sn * dx - c * dy
        d[:, 2] = -c * dx + sn * dy
        d[:, 3] = -sn * dx - c * dy
        d[:, 4] = c * dx - sn * dy
        d[:, 5] = -sn * dx + c * dy
        sup = _support(th[None, :] + base[None, :] + dphi[None, :])
        val = s - d + sup
        imax = np.unravel_index(np.argmax(val), val.shape)
        j = imax[1]
        if j == 0:
            nrm = (c, sn)
        elif j == 1:
            nrm = (sn, -c)
        elif j == 2:
            nrm = (-c, sn)
        elif j == 3:
            nrm = (-sn, -c)
        elif j == 4:
            nrm = (c, -sn)
        else:
            nrm = (-sn, c)
        g = np.zeros_like(z)
        g[0:n] = -nrm[0]
        g[n:2 * n] = -nrm[1]
        g[3 * n] = nrm[0]
        g[3 * n + 1] = nrm[1]
        g[3 * n + 2] = 1.0
        g[3 * n + 3] = -(nrm[0] * dy[imax[0]] - nrm[1] * dx[imax[0]])
        return val.ravel()[0], g

    return evalf, evalf_jac


def _extract(z):
    x = z[0:N]
    y = z[N:2 * N]
    th = z[2 * N:3 * N]
    cx = float(z[3 * N])
    cy = float(z[3 * N + 1])
    s = float(z[3 * N + 2])
    phi = float(z[3 * N + 3])
    inner = [[float(x[i]), float(y[i]), float(np.degrees(th[i]))] for i in range(N)]
    return inner, (cx, cy), s, phi


def _score(z):
    """Outer side length implied by a packing configuration (>= 0)."""
    x = z[0:N]
    y = z[N:2 * N]
    th = z[2 * N:3 * N]
    cx = z[3 * N]
    cy = z[3 * N + 1]
    phi = z[3 * N + 3]
    dx = x - cx
    dy = y - cy
    c = math.cos(phi)
    sn = math.sin(phi)
    cols = [(c, sn), (sn, -c), (-c, sn), (-sn, -c), (c, -sn), (-sn, c)]
    dphi = np.array([0.0, math.pi / 3.0, 2.0 * math.pi / 3.0,
                     math.pi, 4.0 * math.pi / 3.0, 5.0 * math.pi / 3.0])
    base = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    sup = _support(th[None, :] + base[None, :] + dphi[None, :])
    m = 0.0
    for i in range(N):
        for j in range(6):
            val = (cols[j][0] * dx[i] + cols[j][1] * dy[i]) + sup[i, j]
            if val > m:
                m = val
    return m


def _optimize(z0, evalf, evalf_jac, maxiter, tol):
    from scipy.optimize import minimize
    res = minimize(
        evalf, z0, jac=evalf_jac, method="L-BFGS-B",
        bounds=[(None, None)] * (3 * N) + [(0.05, None), (None, None), (None, None)],
        options=dict(maxiter=maxiter, ftol=tol, gtol=1e-8, disp=False),
    )
    return res.x


def construct():
    global _T0
    evalf, evalf_jac = _make_eval()
    z0 = _init_hcp()
    best = z0.copy()
    best_score = _score(best)

    # --- Phase 1: coarse optimization, several restarts with perturbed seeds ---
    for it in range(200):
        if time.time() - _T0 > _DEADLINE:
            break
        if it == 0:
            z = z0.copy()
        else:
            z = best + 0.05 * np.random.randn(3 * N + 4)
            z[3 * N + 2] = max(0.05, best[3 * N + 2] * (1.0 + 0.05 * np.random.randn()))
        try:
            zc = _optimize(z, evalf, evalf_jac, maxiter=300, tol=1e-9)
        except Exception:
            continue
        sc = _score(zc)
        if sc < best_score - 1e-12:
            best = zc.copy()
            best_score = sc

    # --- Phase 2: high-precision polish on the best configuration ---
    for it in range(60):
        if time.time() - _T0 > _DEADLINE:
            break
        try:
            zc = _optimize(best, evalf, evalf_jac, maxiter=400, tol=1e-12)
        except Exception:
            break
        sc = _score(zc)
        if sc < best_score - 1e-12:
            best = zc.copy()
            best_score = sc
        else:
            break

    return _extract(best)


if __name__ == "__main__":
    inner, center, side, angle = construct()
    print("N =", N)
    print("outer side length =", side)
    print("outer center =", center, "rotation =", angle)
    for row in inner:
        print(row)
