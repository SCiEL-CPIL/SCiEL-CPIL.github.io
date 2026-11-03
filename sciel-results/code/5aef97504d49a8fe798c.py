"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_init(n, r):
    """Hexagonal lattice init for n points with circle radius r in the unit disk.

    Returns an (n,2) array of centres, or None if a feasible lattice is not found.
    """
    if r <= 0:
        return None
    dx = 2.0 * r
    dy = dx * np.sqrt(3.0) / 2.0
    k = int(1.0 / dx) + 3
    xs, ys = [], []
    for j in range(-k, k + 1):
        y = j * dy
        if abs(y) > 1.0 - r:
            continue
        xoff = 0.5 * dx * (j % 2)
        for i in range(-k, k + 1):
            x = i * dx + xoff
            if abs(x) > 1.0 - r:
                continue
            if x * x + y * y <= (1.0 - r) ** 2:
                xs.append(x)
                ys.append(y)
    pts = np.column_stack([xs, ys])
    if len(pts) < n:
        return None
    # prefer points closest to the centre (leaves margin at the boundary)
    rad = np.hypot(pts[:, 0], pts[:, 1])
    order = np.argsort(rad, kind="stable")
    return pts[order[:n]]


def _min_dist(pts):
    """Return (min_pair_dist, min_boundary_dist) for an (n,2) array."""
    n = len(pts)
    dx = pts[:, 0:1] - pts[:, 0:1].T
    dy = pts[:, 1:2] - pts[:, 1:2].T
    d2 = dx * dx + dy * dy
    np.fill_diagonal(d2, np.inf)
    min_pair = float(np.sqrt(np.min(d2)))
    min_boundary = float(1.0 - np.max(np.linalg.norm(pts, axis=1)))
    return min_pair, min_boundary


def _score(pts):
    """The packing radius r for an (n,2) array of centres."""
    mp, mb = _min_dist(pts)
    return min(mp / 2.0, mb)


def _is_valid(pts):
    """A configuration is valid if it has N finite, distinct centres inside the disk."""
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return False
    if np.any(np.linalg.norm(pts, axis=1) > 1.0):
        return False
    dx = pts[:, 0:1] - pts[:, 0:1].T
    dy = pts[:, 1:2] - pts[:, 1:2].T
    d2 = dx * dx + dy * dy
    np.fill_diagonal(d2, 0.0)
    if np.min(d2) <= 1e-12:
        return False
    return True


def _soft_min_value(pts, beta):
    """Smooth surrogate for the minimum of {d_ij/2, 1-|c_i|}. Returns (value, pair_d2, norms)."""
    dx = pts[:, 0:1] - pts[:, 0:1].T
    dy = pts[:, 1:2] - pts[:, 1:2].T
    d2 = dx * dx + dy * dy
    np.fill_diagonal(d2, 1e12)
    norms = np.linalg.norm(pts, axis=1)
    b = 1.0 - norms
    # log-sum-exp over the candidate values v = d/2 and b
    vals = np.concatenate([0.5 * np.sqrt(d2), b])
    mx = np.max(vals)
    s = np.sum(np.exp(-beta * (vals - mx)))
    return mx - np.log(s) / beta, d2, norms


def _refine(pts, deadline, beta=30.0):
    """Gradient ascent on the soft-min surrogate with adaptive backtracking step size."""
    n = len(pts)
    best = pts.copy()
    best_score = _score(best)
    step = 0.02
    beta = max(1.0, min(60.0, 16.0 * np.sqrt(N)))  # beta scaled with N
    while time.time() < deadline:
        # --- build the objective value and gradient in one vectorised pass ---
        dx = pts[:, 0:1] - pts[:, 0:1].T
        dy = pts[:, 1:2] - pts[:, 1:2].T
        d2 = dx * dx + dy * dy
        np.fill_diagonal(d2, 1e12)
        d = np.sqrt(d2)
        norms = np.linalg.norm(pts, axis=1)
        b = 1.0 - norms
        vals = np.concatenate([0.5 * d, b])
        mx = np.max(vals)
        w = np.exp(-beta * (vals - mx))
        sw = np.sum(w)
        value = mx - np.log(sw) / beta
        w = w / sw  # softmax weights over all candidate values

        # gradient w.r.t. pts
        g = np.zeros_like(pts)
        wp = w[: n * n].reshape(n, n)
        wb = w[n * n:]
        # pair contribution: d_j = |c_j - c_i|; d/d c_i d_j = -(c_j - c_i)/d_j
        inv = np.where(d > 1e-9, 1.0 / d, 0.0)
        # g_i += sum_j w_ij * (-1/2) * (c_i - c_j)/d_ij  for the pair term (d/2)
        # using row i, col j: diff = c_j - c_i
        g += -0.5 * np.sum(wp * inv * (pts[None, :, :] - pts[:, None, :]), axis=1)
        # boundary contribution: b_i = 1 - |c_i|; d b_i / d c_i = -c_i/|c_i|
        invn = np.where(norms > 1e-9, 1.0 / norms, 0.0)
        g += -wb[:, None] * pts * invn[:, None]

        gnorm = np.linalg.norm(g)
        if gnorm < 1e-12:
            break
        g = g / gnorm

        # --- backtracking line search on the true (non-smooth) radius objective ---
        improved = False
        for _ in range(12):
            cand = pts + step * g
            if np.all(np.linalg.norm(cand, axis=1) < 1.0):  # keep strictly inside
                cs = _score(cand)
                if cs > best_score + 1e-10:
                    pts = cand
                    best = cand.copy()
                    best_score = cs
                    improved = True
                    break
            step *= 0.5
        if not improved:
            step *= 0.5
            if step < 1e-7:
                break
        else:
            step = min(step * 1.15, 0.05)
    return best, best_score


def construct():
    """Centres of N equal circles inside the unit circle, maximizing the common radius."""
    start = time.time()
    deadline = start + 110.0  # safety margin well under the 150s kill limit

    # --- structural initialization: hexagonal lattice at a slightly optimistic radius ---
    r0 = 0.0531542991469  # best-known packing radius (Lai et al. 2021)
    pts = None
    for rr in (r0 * 0.999, r0 * 0.99, r0 * 0.98, r0 * 0.97, r0 * 0.95, r0 * 0.9):
        cand = _hex_init(N, rr)
        if cand is not None and _is_valid(cand):
            pts = cand
            break
    if pts is None:
        # robust fallback: densify the lattice until N valid points exist
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            p = np.array(rows)
            rad = np.hypot(p[:, 0], p[:, 1])
            p = p[rad <= 1.0 - spacing / 2.0]
            if len(p) >= N:
                pts = p[np.argsort(rad, kind="stable")[:N]]
                break
            spacing *= 0.98
            if spacing < 1e-4:
                pts = p[:N]
                break

    best = pts.copy()
    best_score = _score(best)

    # --- refinement machinery: gradient ascent on the smooth surrogate ---
    if time.time() < deadline:
        best, best_score = _refine(best, deadline)

    # --- final safety: guarantee a valid return ---
    if not _is_valid(best):
        best = pts.copy()

    return best
# EVOLVE-BLOCK-END
