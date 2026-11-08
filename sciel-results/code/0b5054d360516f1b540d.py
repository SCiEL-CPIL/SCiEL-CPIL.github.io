"""Sutton-Chen transition-metal cluster global minima - growth-based construction."""
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_PARAMS = {
    "12-6": (12, 6, 144.41),
    "9-6": (9, 6, 39.432),
    "10-8": (10, 8, 34.408),
}
_MIN_DIST = 0.35


def _energy(positions, kind):
    ne, me, c = _PARAMS[kind]
    n = positions.shape[0]
    d = positions[:, None, :] - positions[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", d, d)
    np.fill_diagonal(r2, 1.0)
    r = np.sqrt(r2)
    rn = r ** (-ne)
    rm = r ** (-me)
    np.fill_diagonal(rn, 0.0)
    np.fill_diagonal(rm, 0.0)
    return 0.5 * np.sum(rn) - c * np.sum(np.sqrt(rm))


def _relax(positions, kind, maxiter=2000):
    try:
        e, p = relax(np.asarray(positions, dtype=float), "sutton_chen", kind)
        if np.all(np.isfinite(p)) and np.all(np.isfinite(e)):
            return float(e), np.asarray(p, dtype=float)
    except Exception:
        pass
    return np.inf, np.asarray(positions, dtype=float)


def _valid(p):
    if not np.all(np.isfinite(p)):
        return False
    if p.shape[0] < 2:
        return True
    d = p[:, None, :] - p[None, :, :]
    r = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
    np.fill_diagonal(r, np.inf)
    return r.min() >= _MIN_DIST


def _fcc_positions(n, a=1.0):
    if n <= 0:
        return np.zeros((0, 3))
    R = (n / 4.0) ** (1.0 / 3.0) * a * 1.05
    m = int(np.ceil(R / (a * 0.5)))
    pts = []
    for i in range(-m, m + 1):
        for j in range(-m, m + 1):
            for k in range(-m, m + 1):
                for off in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = a * np.array([i + off[0], j + off[1], k + off[2]])
                    if np.linalg.norm(p) <= R:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) < n:
        return pts
    d = np.linalg.norm(pts, axis=1)
    idx = np.argsort(d)[:n]
    return pts[idx]


def _hcp_positions(n, a=1.0):
    if n <= 0:
        return np.zeros((0, 3))
    R = (n / 4.0) ** (1.0 / 3.0) * a * 1.1
    pts = []
    m = int(np.ceil(R / (a * 0.5))) + 1
    c3 = np.sqrt(3.0) / 2.0
    h = a * np.sqrt(2.0 / 3.0)
    for layer in range(-m, m + 1):
        z = layer * h
        for i in range(-m, m + 1):
            for j in range(-m, m + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * c3 * j
                p = np.array([x, y, z])
                if np.linalg.norm(p) <= R:
                    pts.append(p)
    pts = np.array(pts)
    if len(pts) < n:
        return pts
    d = np.linalg.norm(pts, axis=1)
    idx = np.argsort(d)[:n]
    return pts[idx]


def _mackay_positions(n, a=1.0):
    shells = [
        [(0, 0, 0)],
        [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)],
        [(2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, 2), (0, 0, -2),
         (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0), (1, 0, 1), (1, 0, -1),
         (-1, 0, 1), (-1, 0, -1), (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)],
    ]
    pts = []
    for sh in shells:
        for v in sh:
            pts.append(a * np.array(v, dtype=float))
    pts = np.array(pts)
    d = np.linalg.norm(pts, axis=1)
    idx = np.argsort(d)[:n]
    return pts[idx]


def _random_positions(n, a=0.72, seed=0):
    rng = np.random.default_rng(seed)
    R = a * n ** (1.0 / 3.0) * 0.85
    pts = []
    tries = 0
    while len(pts) < n and tries < 50000:
        tries += 1
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) > R:
            continue
        if pts:
            dd = np.linalg.norm(np.array(pts) - p, axis=1)
            if dd.min() < 0.75 * a:
                continue
        pts.append(p)
    while len(pts) < n:
        p = rng.uniform(-R, R, 3)
        pts.append(p)
    return np.array(pts)


def _growth(core, target, kind, maxiter=1500):
    """Grow a relaxed core to target size by inserting atoms at low-energy sites."""
    pts = list(core)
    a = 0.72
    while len(pts) < target:
        pts_arr = np.array(pts)
        R = a * len(pts_arr) ** (1.0 / 3.0) * 1.1
        # candidate sites: midpoints of close pairs + random points
        cand = []
        d = np.linalg.norm(pts_arr[:, None, :] - pts_arr[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        pair = np.argmin(d, axis=1)
        for i in range(len(pts_arr)):
            mid = 0.5 * (pts_arr[i] + pts_arr[pair[i]])
            cand.append(mid)
        rng = np.random.default_rng(len(pts) * 131 + 7)
        for _ in range(60):
            p = rng.uniform(-R, R, 3)
            if np.linalg.norm(p) <= R:
                cand.append(p)
        cand = np.array(cand)
        best_e, best_p = np.inf, None
        for c in cand:
            trial = np.vstack([pts_arr, c])
            e, _ = _relax(trial, kind, maxiter=maxiter)
            if e < best_e:
                best_e, best_p = e, trial
        if best_p is None:
            p = np.array(pts[-1]) + a * np.array([0.3, 0.3, 0.3])
            best_p = np.vstack([pts_arr, p])
        pts = list(best_p)
    return np.array(pts)


def construct(instance):
    deadline = time.time() + 250.0
    n = instance["n"]
    kind = instance["kind"]

    cores = [
        _fcc_positions(n, 1.0),
        _hcp_positions(n, 1.0),
        _mackay_positions(n, 1.0),
        _random_positions(n, 0.72, seed=n),
        _random_positions(n, 0.72, seed=n + 1000),
    ]

    best_e, best_p = np.inf, None
    for core in cores:
        if time.time() > deadline:
            break
        c = np.asarray(core, dtype=float)
        if c.shape[0] < n:
            c = _growth(c, n, kind)
        elif c.shape[0] > n:
            c = c[:n]
        e, p = _relax(c, kind, maxiter=2000)
        if _valid(p) and e < best_e:
            best_e, best_p = e, p

    if best_p is None:
        best_p = _random_positions(n, 0.72, seed=n)

    # final high-precision polish
    if time.time() < deadline - 20:
        e, p = _relax(best_p, kind, maxiter=6000)
        if _valid(p) and e < best_e:
            best_e, best_p = e, p

    if not _valid(best_p):
        # last-resort valid fallback
        best_p = _fcc_positions(n, 1.0)
        if best_p.shape[0] < n:
            best_p = _random_positions(n, 0.72, seed=n)

    return np.asarray(best_p, dtype=float)
