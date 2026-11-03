"""Multi-stage optimizer for 300 equal circles in the unit disk."""
import time
import numpy as np

N = 300
DEADLINE = 150.0
START = time.time()

def _elapsed():
    return time.time() - START

def _score(pts):
    """Return the valid common radius (min of pair-dist/2 and 1-|c|)."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # boundary
    bound = 1.0 - np.linalg.norm(pts, axis=1)
    r_bound = float(bound.min())
    # pairwise via chunked to control memory
    best = 1e9
    block = 64
    for s in range(0, N, block):
        e = min(s + block, N)
        d = np.linalg.norm(pts[s:e, None, :] - pts[None, :, :], axis=2)
        # zero out self
        np.fill_diagonal(d, 1e9)
        best = min(best, float(d.min()))
    return min(r_bound, best / 2.0)

def _hex_seed(r0):
    """Build a hexagonal lattice of points inside the unit disk, pick N nearest to center."""
    spacing = 2.0 * r0
    # oversample grid
    k = int(1.0 / spacing) + 3
    rows = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            rows.append((x, y))
    pts = np.array(rows, dtype=np.float64)
    rad = np.linalg.norm(pts, axis=1)
    keep = rad <= 1.0 - r0 - 1e-9
    pts = pts[keep]
    if len(pts) < N:
        # fall back: scale down spacing until enough
        for _ in range(200):
            r0 *= 0.995
            spacing = 2.0 * r0
            k = int(1.0 / spacing) + 3
            rows = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    rows.append((x, y))
            pts = np.array(rows, dtype=np.float64)
            rad = np.linalg.norm(pts, axis=1)
            keep = rad <= 1.0 - r0 - 1e-9
            pts = pts[keep]
            if len(pts) >= N:
                break
    # choose N points: nearest to origin (densest core)
    rad = np.linalg.norm(pts, axis=1)
    idx = np.argsort(rad, kind="stable")[:N]
    return pts[idx].copy(), r0

def _softmin_radius(pts, beta):
    """Smooth surrogate: soft-min of (pair_dist/2) and (1-|c|) using log-sum-exp."""
    # boundary term
    bound = 1.0 - np.linalg.norm(pts, axis=1)
    # pairwise soft-min
    block = 64
    # We want soft min over all pair distances/2 and boundary.
    # Use log-sum-exp in the "min" direction: min ≈ -logsumexp(-x)/beta form.
    # Build list of values.
    vals = []
    for s in range(0, N, block):
        e = min(s + block, N)
        d = np.linalg.norm(pts[s:e, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(d, 1e9)
        vals.append(d / 2.0)
    vals = np.concatenate(vals, axis=0)
    vals = np.concatenate([vals, bound])
    # soft min via log-sum-exp of -vals
    m = vals.min()
    z = -beta * (vals - m)
    z = z - z.max()
    lse = np.log(np.exp(z).sum())
    softmin = m - lse / beta
    return float(softmin)

def _softmin_grad(pts, beta):
    """Gradient of the soft-min surrogate w.r.t. pts (for ascent)."""
    bound = 1.0 - np.linalg.norm(pts, axis=1)
    # gradient of boundary term: d(1-|c|)/dc = -c/|c|
    rnorm = np.linalg.norm(pts, axis=1, keepdims=True)
    rnorm[rnorm < 1e-12] = 1e-12
    g_bound = -pts / rnorm
    # pairwise soft-min gradient
    block = 64
    # collect weights via log-sum-exp
    # We'll accumulate gradient.
    g = np.zeros_like(pts)
    # Build all values and their index mapping is heavy; instead do per-block
    # but log-sum-exp needs global normalization. Compute global m and weights per block.
    # First pass: compute global min m over all candidate values.
    m = bound.min()
    for s in range(0, N, block):
        e = min(s + block, N)
        d = np.linalg.norm(pts[s:e, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(d, 1e9)
        m = min(m, float((d/2.0).min()))
    # second pass: accumulate weighted contributions
    # weight for value v: exp(-beta*(v - m)) / Z ; Z = sum over all
    # We compute Z and per-block sums.
    block_weights = []
    Z = 0.0
    for s in range(0, N, block):
        e = min(s + block, N)
        d = np.linalg.norm(pts[s:e, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(d, 1e9)
        v = d / 2.0
        w = np.exp(-beta * (v - m))
        block_weights.append((s, e, v, w))
        Z += float(w.sum())
    # boundary weight
    wb = np.exp(-beta * (bound - m))
    Z += float(wb.sum())
    # gradient of softmin = - (1/beta) * sum_i w_i * grad(v_i) / Z  ... careful with sign
    # softmin = m - (1/beta) log Z, where Z = sum exp(-beta (v - m))
    # d softmin / d pts = -(1/beta) * d log Z / d pts = -(1/beta) * (1/Z) sum_i exp(-beta(v_i-m)) dv_i/dpts
    # For boundary: v = 1-|c|, dv/dc = -c/|c|
    g = g + (-(1.0/beta) / Z) * (wb[:, None] * g_bound)
    for (s, e, v, w) in block_weights:
        # dv/dpts for pair (i,j) with i in [s,e), j any: d(|c_i-c_j|/2)/dc_i = (c_i-c_j)/(2|c_i-c_j|)
        # accumulate
        gi = np.zeros((e - s, 2))
        for jj in range(N):
            diff = pts[s:e, :] - pts[jj:jj+1, :]
            dist = np.linalg.norm(diff, axis=1, keepdims=True)
            dist[dist < 1e-12] = 1e-12
            grad = diff / (2.0 * dist)
            # weight row for this pair
            wi = w[:, jj]
            gi += wi[:, None] * grad
        # add contribution to global g at rows s:e
        g[s:e, :] += (-(1.0/beta) / Z) * gi
    return g

def _project_inside(pts, r):
    """Push any point outside (1 - r) back to the boundary."""
    rad = np.linalg.norm(pts, axis=1, keepdims=True)
    limit = 1.0 - r
    out = rad > limit
    if out.any():
        pts = pts.copy()
        pts[out] = pts[out] * (limit / rad[out])[:, None]
    return pts

def _optimize(pts, r0, beta_start, beta_end, steps, lr, deadline):
    """Annealed soft-min ascent with Hertzian-ish repulsion blended in."""
    best = _score(pts)
    best_pts = pts.copy()
    beta = beta_start
    for it in range(steps):
        if _elapsed() > deadline:
            break
        # blend annealing
        frac = it / max(1, steps - 1)
        beta = beta_start * (beta_end / beta_start) ** frac
        # gradient ascent on soft-min radius
        g = _softmin_grad(pts, beta)
        # also add a repulsion to avoid coincident points (Hertzian-like)
        # compute nearest-neighbor repulsion
        block = 64
        rep = np.zeros_like(pts)
        for s in range(0, N, block):
            e = min(s + block, N)
            d = np.linalg.norm(pts[s:e, None, :] - pts[None, :, :], axis=2)
            np.fill_diagonal(d, 1e9)
            # repulsion ~ 1/d^2 pointing apart
            inv = np.where(d < 2*r0, 1.0 / np.maximum(d, 1e-6)**2, 0.0)
            diff = pts[s:e, None, :] - pts[None, :, :]
            rep[s:e] += (inv * diff).sum(axis=1)
        grad = g + 0.05 * rep
        # step
        pts = pts + lr * grad
        # project inside
        pts = _project_inside(pts, r0)
        # evaluate
        sc = _score(pts)
        if sc > best:
            best = sc
            best_pts = pts.copy()
    return best_pts, best

def construct():
    deadline = START + DEADLINE - 10.0  # 10s safety margin
    # Stage 1: hexagonal seed at target density
    r_target = 0.05315
    pts, r0 = _hex_seed(r_target)
    best = _score(pts)
    best_pts = pts.copy()
    if _elapsed() > DEADLINE - 20:
        return best_pts

    # Stage 2: initial optimization
    pts2, sc2 = _optimize(best_pts, r0, beta_start=40.0, beta_end=200.0,
                          steps=200, lr=0.002, deadline=deadline)
    if sc2 > best:
        best = sc2
        best_pts = pts2
    if _elapsed() > DEADLINE - 20:
        return best_pts

    # Stage 3: random restarts with perturbation
    rng = np.random.default_rng(12345)
    restarts = 0
    while _elapsed() < deadline - 15:
        restarts += 1
        # perturb current best
        noise_scale = 0.002 + 0.001 * (restarts % 3)
        pts_p = best_pts + rng.normal(0.0, noise_scale, size=(N, 2))
        pts_p = _project_inside(pts_p, r0)
        sc_p = _score(pts_p)
        if sc_p > best:
            best = sc_p
            best_pts = pts_p
        # short optimize from perturbed
        pts3, sc3 = _optimize(pts_p, r0, beta_start=60.0, beta_end=250.0,
                              steps=80, lr=0.0015, deadline=deadline)
        if sc3 > best:
            best = sc3
            best_pts = pts3

    # Stage 4: final high-precision polish
    if _elapsed() < deadline:
        pts4, sc4 = _optimize(best_pts, r0, beta_start=200.0, beta_end=800.0,
                              steps=120, lr=0.0005, deadline=deadline)
        if sc4 > best:
            best = sc4
            best_pts = pts4

    # final validity guard
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 2):
        # fallback to hex seed
        return _hex_seed(r_target)[0]
    return best_pts
