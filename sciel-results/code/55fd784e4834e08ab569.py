import numpy as np

N = 12
D = np.sqrt(3.0)            # min center distance for same-oriented unit hexagons (touch)
GAP = 1e-5                  # tiny safety gap to guarantee clear disjointness
MIN_DIST = D + GAP


def _support(u):
    """Support function of a unit regular hexagon (circumradius 1, orientation 0) in direction u."""
    a = np.arctan2(u[1], u[0])
    base = np.deg2rad(60.0) * np.arange(6)
    return float(np.max(np.cos(a - base)))


def _s_for_theta(theta, centers):
    """Minimal outer side length for fixed rotation theta (radians), optimal translation."""
    k = np.arange(6)
    ang = theta + np.deg2rad(30.0) + np.deg2rad(60.0) * k
    cx = np.cos(ang); sy = np.sin(ang)
    best = 0.0
    for kk in range(6):
        nx, ny = cx[kk], sy[kk]
        h_pos = _support((nx, ny))
        h_neg = _support((-nx, -ny))
        m_pos = float(np.max(centers[:, 0] * nx + centers[:, 1] * ny + h_pos))
        m_neg = float(np.max(centers[:, 0] * (-nx) + centers[:, 1] * (-ny) + h_neg))
        v = m_pos + m_neg
        if v > best:
            best = v
    return best / D


def _objective(centers, thetas):
    m = 1e9
    for t in thetas:
        s = _s_for_theta(t, centers)
        if s < m:
            m = s
    return m


def _valid(centers):
    md2 = MIN_DIST * MIN_DIST
    for i in range(N):
        xi, yi = centers[i, 0], centers[i, 1]
        for j in range(i + 1, N):
            dx = xi - centers[j, 0]
            dy = yi - centers[j, 1]
            if dx * dx + dy * dy < md2:
                return False
    return True


def _hex_lattice(i, j):
    return np.array([D * (i + j / 2.0), D * (np.sqrt(3.0) / 2.0) * j])


def _make_cluster(kind):
    if kind == 0:      # 4x3 parallelogram
        pts = [_hex_lattice(i, j) for j in (-1, 0, 1) for i in (0, 1, 2, 3)]
    elif kind == 1:    # 5,4,3 hex-ish
        pts = ([_hex_lattice(i, -1) for i in (0, 1, 2, 3, 4)] +
               [_hex_lattice(i, 0) for i in (0, 1, 2, 3)] +
               [_hex_lattice(i, 1) for i in (0, 1, 2)])
    else:              # 4,4,4 centered
        pts = ([_hex_lattice(i, -1) for i in (-1, 0, 1, 2)] +
               [_hex_lattice(i, 0) for i in (-2, -1, 0, 1)] +
               [_hex_lattice(i, 1) for i in (-1, 0, 1, 2)])
    return np.array(pts, dtype=float)


def _local_search(centers, thetas, iters, step0, step1, seed):
    rng = np.random.default_rng(seed)
    cur = centers.astype(float).copy()
    cur_obj = _objective(cur, thetas)
    best = cur.copy(); best_obj = cur_obj
    T0, T1 = 0.04, 0.0008
    for it in range(iters):
        frac = it / max(1, iters - 1)
        step = step0 + (step1 - step0) * frac
        T = T0 + (T1 - T0) * frac
        i = int(rng.integers(N))
        cand = cur.copy()
        cand[i] += rng.standard_normal(2) * step
        if not _valid(cand):
            continue
        obj = _objective(cand, thetas)
        if obj <= cur_obj or rng.random() < np.exp(-(obj - cur_obj) / max(T, 1e-9)):
            cur = cand; cur_obj = obj
            if obj < best_obj:
                best = cur.copy(); best_obj = obj
    return best, best_obj


def _best_theta(centers, thetas):
    bt, bs = thetas[0], 1e9
    for t in thetas:
        s = _s_for_theta(t, centers)
        if s < bs:
            bs, bt = s, t
    return bt, bs


def _compute_center(theta, centers, s):
    k = np.arange(6)
    ang = theta + np.deg2rad(30.0) + np.deg2rad(60.0) * k
    normals = np.column_stack([np.cos(ang), np.sin(ang)])
    mids = []
    for kk in range(6):
        nk = normals[kk]
        h_pos = _support(nk); h_neg = _support(-nk)
        m_pos = float(np.max(centers @ nk + h_pos))
        m_neg = float(np.max(centers @ (-nk) + h_neg))
        L = m_pos - s * D / 2.0
        U = -m_neg + s * D / 2.0
        mids.append((L + U) / 2.0)
    c, *_ = np.linalg.lstsq(normals, np.array(mids), rcond=None)
    return c


def _verify(theta, centers, center, s):
    k = np.arange(6)
    ang = theta + np.deg2rad(30.0) + np.deg2rad(60.0) * k
    normals = np.column_stack([np.cos(ang), np.sin(ang)])
    lim = s * D / 2.0 + 1e-6
    for i in range(N):
        d = centers[i] - center
        for kk in range(6):
            nk = normals[kk]
            if float(nk @ d) + _support(nk) > lim:
                return False
    return True


def _fallback():
    inner = [[(i - (N - 1) / 2.0) * D, 0.0, 0.0] for i in range(N)]
    return inner, (0.0, 0.0), 30.0, 0.0


def construct():
    try:
        thetas_coarse = np.linspace(0.0, np.deg2rad(60.0), 20, endpoint=False)
        thetas_fine = np.linspace(0.0, np.deg2rad(60.0), 120, endpoint=False)

        best_centers = None
        best_val = 1e9
        for kind in range(3):
            base = _make_cluster(kind)
            for seed in (0, 1):
                rng = np.random.default_rng(1000 + kind * 10 + seed)
                centers = base + rng.standard_normal((N, 2)) * 0.05
                if not _valid(centers):
                    centers = base.copy()
                centers, _ = _local_search(centers, thetas_coarse, 1200, 0.5, 0.05, seed)
                centers, _ = _local_search(centers, thetas_fine, 400, 0.2, 0.02, seed + 7)
                theta, s = _best_theta(centers, thetas_fine)
                if s < best_val:
                    best_val = s
                    best_centers = centers.copy()
                    best_theta = theta

        theta, s = _best_theta(best_centers, thetas_fine)
        center = _compute_center(theta, best_centers, s)
        # safety: grow side until verified
        for _ in range(60):
            if _verify(theta, best_centers, center, s):
                break
            s += 0.01
            center = _compute_center(theta, best_centers, s)

        inner = [[float(x), float(y), 0.0] for (x, y) in best_centers]
        return inner, (float(center[0]), float(center[1])), float(s), float(np.rad2deg(theta))
    except Exception:
        return _fallback()
