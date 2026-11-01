"""Rigid-body repulsion packing for 200 equal circles in the unit square."""
import time

N = 200
T_START = time.time()
DEADLINE = T_START + 115.0  # safety margin before the 120 s kill (150 s wall budget)


def _time_left():
    return DEADLINE - time.time()


def _compute_score(c):
    """Largest common radius for centres c (N,2) inside [-0.5,0.5]^2."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # wall clearance
    d = np.minimum.reduce([
        np.min(0.5 - c[:, 0]),
        np.min(c[:, 0] + 0.5),
        np.min(0.5 - c[:, 1]),
        np.min(c[:, 1] + 0.5),
    ])
    # pairwise
    diff = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    pmin = np.min(dist) / 2.0
    return float(min(d, pmin))


def _grid_init():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _simulate(c, dt=0.0004, nsteps=12000, damping=0.995):
    """Velocity-Verlet with a short-range repulsive potential + wall springs."""
    c = c.astype(np.float64).copy()
    v = np.zeros_like(c)
    # precompute pair index list (i<j) for speed
    ii, jj = np.triu_indices(N, k=1)
    for step in range(nsteps):
        if step % 200 == 0 and _time_left() < 8.0:
            break
        # current displacements
        d = c[ii, :] - c[jj, :]
        r = np.sqrt(np.sum(d * d, axis=1))
        r = np.maximum(r, 1e-9)
        # repulsive force: strong within a window, zero beyond
        R = 0.06
        mask = r < R
        f = np.zeros_like(d)
        if np.any(mask):
            rm = r[mask]
            dm = d[mask]
            # soft repulsion ~ (1 - r/R)^2 direction, scaled
            scale = 1.0 / (rm * rm)
            fm = (dm / rm) * (1.0 - rm / R) * 8.0
            # scatter back
            idx = np.where(mask)[0]
            f[idx] = fm
        acc = np.zeros_like(c)
        np.add.at(acc, ii, -f)
        np.add.at(acc, jj, f)
        # wall springs
        for ax in (0, 1):
            lo = c[:, ax] + 0.5
            hi = 0.5 - c[:, ax]
            # push inward when within margin
            mlo = np.clip(0.06 - lo, 0.0, None)
            mhi = np.clip(0.06 - hi, 0.0, None)
            acc[:, ax] += mlo * 40.0 - mhi * 40.0
        # velocity Verlet
        v = v * damping + acc * dt
        c = c + v * dt
        # hard clamp inside square
        c[:, 0] = np.clip(c[:, 0], -0.5, 0.5)
        c[:, 1] = np.clip(c[:, 1], -0.5, 0.5)
    return c


def _polish(c, max_iter=600, tol=1e-10):
    """Local optimization: maximize min distance via subgradient / steepest-ascent on soft-min."""
    c = c.astype(np.float64).copy()
    ii, jj = np.triu_indices(N, k=1)
    best = None
    for it in range(max_iter):
        if _time_left() < 6.0:
            break
        d = c[ii, :] - c[jj, :]
        r = np.sqrt(np.sum(d * d, axis=1))
        r = np.maximum(r, 1e-12)
        # wall distances
        w = np.array([np.min(0.5 - c[:, 0]), np.min(c[:, 0] + 0.5),
                      np.min(0.5 - c[:, 1]), np.min(c[:, 1] + 0.5)])
        allv = np.concatenate([r / 2.0, w])
        mval = np.min(allv)
        if best is None or mval > best[0]:
            best = (mval, c.copy())
        if it > 20 and mval - best[0] < 1e-13:
            break
        # gradient of soft-min (smooth surrogate) w.r.t. each centre
        g = np.zeros_like(c)
        # pair contribution
        beta = 30.0 / max(mval, 1e-9)
        wexp = np.exp(-beta * (r / 2.0 - mval))
        wsum = np.sum(wexp)
        if wsum > 0:
            wexp = wexp / wsum
            contrib = wexp[:, None] * (d / r[:, None]) / r
            np.add.at(g, ii, -contrib)
            np.add.at(g, jj, contrib)
        # wall contribution
        wexp2 = np.exp(-beta * (w - mval))
        wsum2 = np.sum(wexp2)
        if wsum2 > 0:
            wexp2 = wexp2 / wsum2
            # which point attains each wall min
            widx = np.array([np.argmin(0.5 - c[:, 0]), np.argmax(c[:, 0]),
                             np.argmin(0.5 - c[:, 1]), np.argmax(c[:, 1])])
            wdir = np.array([(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)])
            for k in range(4):
                g[widx[k]] += wexp2[k] * wdir[k]
        # normalize gradient
        gn = np.sqrt(np.sum(g * g))
        if gn > 0:
            g = g / gn
        # adaptive step
        step = 2e-4
        # line search
        for _ in range(30):
            cand = c + g * step
            cand[:, 0] = np.clip(cand[:, 0], -0.5, 0.5)
            cand[:, 1] = np.clip(cand[:, 1], -0.5, 0.5)
            # quick eval
            dd = cand[ii, :] - cand[jj, :]
            rr = np.sqrt(np.sum(dd * dd, axis=1))
            np.fill_diagonal_safe = None
            rr2 = np.minimum.reduce([
                np.min(0.5 - cand[:, 0]), np.min(cand[:, 0] + 0.5),
                np.min(0.5 - cand[:, 1]), np.min(cand[:, 1] + 0.5),
                np.min(rr) / 2.0,
            ])
            if rr2 > mval + 1e-12:
                c = cand
                break
            step *= 0.5
    return best[1] if best is not None else c


def construct():
    import numpy as np
    np.random.seed(0)
    c = _grid_init()
    # small random jitter to break grid symmetry
    c = c + np.random.randn(N, 2) * 1e-4
    c = _simulate(c)
    c = _polish(c)
    # a couple more polish passes if time remains
    while _time_left() > 12.0:
        c2 = _polish(c, max_iter=200)
        if _compute_score(c2) <= _compute_score(c) + 1e-14:
            break
        c = c2
    return c


import numpy as np
