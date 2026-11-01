"""Pack 50 equal circles in the unit disk via gradient ascent on the radius."""
import time
import numpy as np

N = 50
DEADLINE = time.time() + 110.0  # safety margin before the 120s (150s hard) kill

def _score(pts):
    """Common radius: min of half pairwise distances and margin to boundary."""
    d = np.hypot(pts[:, 0] - pts[1:, 0], pts[:, 1] - pts[1:, 1])
    min_dist = d.min()
    margin = 1.0 - np.hypot(pts[:, 0], pts[:, 1]).max()
    return 0.5 * min_dist, margin

def _soft_min_obj_grad(pts, beta):
    """Smooth surrogate for min(half-pair-dist, margin). Returns (value, grad).
    Uses log-sum-exp (soft-min) over the 'distance' terms and the 'margin' terms."""
    n = len(pts)
    iu = np.triu_indices(n, k=1)
    dx = pts[iu[0], 0] - pts[iu[1], 0]
    dy = pts[iu[0], 1] - pts[iu[1], 1]
    dist = np.hypot(dx, dy)
    dist = np.where(dist < 1e-12, 1e-12, dist)
    half = 0.5 * dist
    margin = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    margin = np.where(margin < 1e-12, 1e-12, margin)

    # soft-min over all candidate radii (pair halves + margins)
    allvals = np.concatenate([half, margin])
    maxv = allvals.max()
    # soft-min = -soft-max of -x
    neg = -allvals
    m = neg.max()
    s = np.exp(neg - m)
    ss = s.sum()
    w = s / ss
    softmin = -(m + np.log(ss))

    # gradient of soft-min w.r.t. each value: d(softmin)/d(val_k) = -w_k
    g = np.zeros((n, 2))
    # pair terms
    wh = w[:len(half)]
    # d(half)/d(pts)
    # half_k = 0.5*|d_k|, d(half)/d(pts_i) = 0.5 * dvec/|dvec|
    inv = wh * (1.0 / dist) * 0.5  # 0.5 * w_k / dist_k
    for k in range(len(half)):
        a, b = iu[0][k], iu[1][k]
        g[a, 0] += -inv[k] * dx[k]
        g[a, 1] += -inv[k] * dy[k]
        g[b, 0] += inv[k] * dx[k]
        g[b, 1] += inv[k] * dy[k]
    # margin terms
    wm = w[len(half):]
    r = np.hypot(pts[:, 0], pts[:, 1])
    r = np.where(r < 1e-12, 1e-12, r)
    # margin = 1 - r, d(margin)/d(pts) = -pts/r
    for k in range(n):
        g[k, 0] += -wm[k] * (-pts[k, 0] / r[k])
        g[k, 1] += -wm[k] * (-pts[k, 1] / r[k])
    return softmin, g

def _ring_init(k):
    """Init with k concentric rings of points, symmetric."""
    pts = []
    # place N points on k rings
    import math
    for ring in range(k):
        if ring == 0:
            pts.append([0.0, 0.0])
        else:
            cnt = N - 1 - sum(len([0]) for _ in range(ring))
            cnt = N - 1 - (k - 1 - ring) * 6  # rough
            cnt = max(6, N - 1 - (k - 1 - ring) * 6)
            r = 0.75 * ring / max(1, k - 1)
            for c in range(cnt):
                ang = 2 * math.pi * c / cnt + math.pi * ring / k
                pts.append([r * math.cos(ang), r * math.sin(ang)])
            if len(pts) >= N:
                break
    pts = np.array(pts[:N])
    return pts

def _hex_init():
    """Hexagonal lattice cut to unit disk, sorted by radius."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def _optimize(pts, beta_start, beta_end, max_iter, step0):
    """Gradient ascent on the soft-min radius surrogate, annealing beta."""
    best = _score(pts)[0]
    best_pts = pts.copy()
    beta = beta_start
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        val, g = _soft_min_obj_grad(pts, beta)
        # anneal beta
        beta = beta_start * (beta_end / beta_start) ** (it / max(1, max_iter))
        step = step0 * (1.0 - 0.5 * it / max(1, max_iter))
        pts = pts + step * g
        # project inside disk
        r = np.hypot(pts[:, 0], pts[:, 1])
        over = r > 1.0
        if over.any():
            pts[over] = pts[over] * (1.0 / r[over])
        cur, _ = _score(pts)
        if cur > best:
            best = cur
            best_pts = pts.copy()
    return best_pts, best

def construct():
    start = time.time()
    # base init
    base = _hex_init()
    b0, s0 = _score(base)
    best_pts, best = _optimize(base.copy(), 8.0, 200.0, 400, 0.02)

    # restarts with perturbation and ring inits
    rng = np.random.default_rng(12345)
    for attempt in range(40):
        if time.time() > DEADLINE - 5.0:
            break
        # choose init
        if attempt % 3 == 0:
            init = _ring_init(1 + (attempt // 3) % 4)
        elif attempt % 3 == 1:
            init = _hex_init() + rng.normal(0, 0.05, (N, 2))
        else:
            init = best_pts + rng.normal(0, 0.04, (N, 2))
        # project init inside
        r = np.hypot(init[:, 0], init[:, 1])
        over = r > 0.99
        if over.any():
            init[over] = init[over] * (0.99 / r[over])
        beta_s = 6.0 + 4.0 * (attempt % 5)
        step0 = 0.025 - 0.001 * (attempt % 4)
        pts, sc = _optimize(init, beta_s, 220.0, 350, step0)
        if sc > best:
            best = sc
            best_pts = pts.copy()

    # final high-precision polish on the best
    if time.time() < DEADLINE - 8.0:
        pts, sc = _optimize(best_pts.copy(), 30.0, 400.0, 300, 0.004)
        if sc > best:
            best = sc
            best_pts = pts.copy()

    # ensure validity: strictly inside
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    over = r >= 1.0
    if over.any():
        best_pts[over] = best_pts[over] * (0.999 / r[over])
    return best_pts
