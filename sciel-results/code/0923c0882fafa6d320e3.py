"""Circle packing in a unit square: repulsion-based optimizer with annealing + polish."""
import numpy as np
from math import log, exp, sqrt
import time

N = 200

# EVOLVE-BLOCK-START
def _score(pts):
    """Raw score: min(pairwise dist/2, wall clearance)."""
    d = np.abs(pts[:, None, :] - pts[None, :, :])
    d = np.sqrt(d[:, :, 0] ** 2 + d[:, :, 1] ** 2)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    wall = 0.5 - np.abs(pts).max(axis=0).max()
    return min(min_pair, wall)


def _repulsion_grad(pts, k, wall_k, alpha):
    """
    Gradient of the objective surrogate:
      sum_{i<j} w_ij * d_ij  (soft repulsion, w grows when d small)
    + wall terms
    We maximize this; use gradient ascent.
    """
    # pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]  # (N,N,2)
    d = np.sqrt((diff ** 2).sum(-1))          # (N,N)
    np.fill_diagonal(d, 1.0)
    # weight: repulsion strength grows as d shrinks below 1
    # Use w = exp(k * (1 - d)) / (d + eps)  -> strong when d<1
    eps = 1e-9
    w = np.exp(np.clip(k * (1.0 - d), -30, 30)) / (d + eps)
    np.fill_diagonal(w, 0.0)
    # grad of sum w_ij * d_ij w.r.t. pts_i = sum_j w_ij * (pts_i - pts_j)/d_ij
    # = sum_j w_ij * diff_ij / d_ij
    unit = diff / (d[..., None] + eps)
    grad = (w[..., None] * unit).sum(0)  # (N,2)

    # wall repulsion: push away from edges when close
    # clearance per axis
    left = pts[:, 0] + 0.5
    right = 0.5 - pts[:, 0]
    bottom = pts[:, 1] + 0.5
    top = 0.5 - pts[:, 1]
    # gradient contribution: -wall_k * exp(wall_k * clearance) if clearance small
    def wall_grad(clear):
        # force = wall_k * exp(-wall_k * clear) pointing inward
        # d/dx of -exp(-k*clear) where clear = 0.5 - x => +k*exp(-k*clear)
        return wall_k * np.exp(-wall_k * np.clip(clear, 0, 5))
    gx = np.where(pts[:, 0] > 0, -wall_grad(right), wall_grad(left))
    gy = np.where(pts[:, 1] > 0, -wall_grad(top), wall_grad(bottom))
    grad[:, 0] += gx
    grad[:, 1] += gy
    return grad


def _repulsion_value(pts, k):
    """Value of the soft objective (for tracking)."""
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    w = np.exp(np.clip(k * (1.0 - d), -30, 30)) / (d + 1e-9)
    np.fill_diagonal(w, 0.0)
    return (w * d).sum()


def _project(pts):
    """Clamp to square (with tiny margin) to keep valid."""
    m = 0.5 - 1e-9
    pts = np.clip(pts, -m, m)
    return pts


def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin under 150s
    m = 15  # grid m x m = 225 >= 200
    # Initial: grid, then take first 200
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].astype(np.float64)
    # small random jitter to break symmetry
    rng = np.random.default_rng(12345)
    pts = pts + rng.normal(0, 0.002, pts.shape)
    pts = _project(pts)

    best_pts = pts.copy()
    best_score = _score(pts)

    # Annealing schedule
    k_start = 8.0
    k_end = 40.0
    wall_k = 60.0
    n_iter = 4000
    lr0 = 0.004
    lr_end = 0.00005

    for it in range(n_iter):
        if time.time() - t0 > DEADLINE:
            break
        frac = it / max(1, n_iter - 1)
        k = k_start + (k_end - k_start) * frac
        lr = lr0 * (lr_end / lr0) ** frac
        # momentum
        if it == 0:
            vel = np.zeros_like(pts)
        grad = _repulsion_grad(pts, k, wall_k, lr)
        vel = 0.85 * vel + lr * grad
        pts = pts + vel
        pts = _project(pts)
        # track best
        if it % 50 == 0:
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()

    # Final polish: steepest ascent with line search on exact score, small steps
    if time.time() - t0 < DEADLINE - 10:
        pts = best_pts.copy()
        for p in range(3000):
            if time.time() - t0 > DEADLINE - 5:
                break
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
            # subgradient of min: pick the active constraint
            diff = pts[:, None, :] - pts[None, :, :]
            d = np.sqrt((diff ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            min_pair = d.min() / 2.0
            wall = 0.5 - np.abs(pts).max()
            if min_pair <= wall:
                # active pair
                idx = np.unravel_index(np.argmin(d), d.shape)
                i, j = idx
                v = (pts[i] - pts[j]) / (d[i, j] + 1e-12)
                g = np.zeros_like(pts)
                g[i] += 0.5 * v
                g[j] -= 0.5 * v
            else:
                # active wall
                g = np.zeros_like(pts)
                ax = np.abs(pts).argmax(axis=1)
                # find the point(s) achieving max abs
                for i in range(N):
                    a = np.abs(pts[i]).argmax()
                    g[i, a] = -0.5 * np.sign(pts[i, a])
            g /= (np.linalg.norm(g) + 1e-12)
            # line search
            step = 0.002
            for _ in range(6):
                cand = _project(pts + step * g)
                cs = _score(cand)
                if cs > s:
                    pts = cand
                    s = cs
                    break
                step *= 0.5
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Ensure validity: no coincident, inside
    best_pts = _project(best_pts)
    # final dedup safety
    return best_pts
# EVOLVE-BLOCK-END
