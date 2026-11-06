"""Rewritten program: Fibonacci-disk init + force relaxation + polish for N=300 circle packing."""
import time
import numpy as np

N = 300
DEADLINE = 140.0  # safety margin under 150s wall-clock


def _score(pts):
    """Largest common radius for a set of centres (validity-checked)."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # distance to boundary
    dist_b = 1.0 - np.linalg.norm(pts, axis=1)
    if np.any(dist_b < 0.0):
        return 0.0
    # pairwise distances via chunked computation to bound memory
    n = pts.shape[0]
    min_pair = np.inf
    chunk = 64
    for i in range(0, n, chunk):
        block = pts[i:i + chunk]
        d = np.linalg.norm(block[:, None, :] - pts[None, :, :], axis=2)
        # zero out self and lower triangle handled by taking min over all but self
        np.fill_diagonal(d, np.inf)
        min_pair = min(min_pair, float(d.min()))
    return min(min_pair / 2.0, float(dist_b.min()))


def _init_fibonacci():
    """Evenly spread N points in the unit disk using a Fibonacci/sunflower map."""
    n = N
    idx = np.arange(n, dtype=np.float64)
    # radius: sqrt of evenly spaced area fractions (slightly inset)
    r = np.sqrt((idx + 0.5) / n) * 0.98
    theta = idx * np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))


def construct():
    t0 = time.time()
    best = _init_fibonacci()
    best_s = _score(best)

    def remaining():
        return DEADLINE - (time.time() - t0)

    # --- Main force-relaxation loop ---
    # Adaptive repulsion strength that tracks the current best radius.
    pts = best.copy()
    n = N

    # Precompute nothing heavy; do a few warm-up sweeps to establish a repulsion scale.
    # Estimate initial minimum distance to set repulsion.
    def min_pair_dist(p):
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        return float(d.min())

    # Warm-up: a few fixed-step iterations to get a reasonable radius estimate.
    step = 1e-4
    for _ in range(400):
        if remaining() < 20.0:
            break
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        # repulsion: push apart when closer than 2*r_est
        r_est = best_s if best_s > 1e-9 else 0.05
        dmin = 2.0 * r_est
        # force magnitude ~ 1/d^2 clipped
        inv = np.zeros_like(d)
        mask = d < dmin
        inv[mask] = 1.0 / (d[mask] ** 2 + 1e-12)
        # direction vectors
        diff = pts[None, :, :] - pts[:, None, :]  # diff[i,j] = pts[j]-pts[i]
        # normalize
        d_safe = np.where(d > 1e-12, d, 1.0)
        diff_n = diff / d_safe[:, :, None]
        F = inv[:, :, None] * diff_n
        F = F.sum(axis=1)
        # boundary repulsion
        rn = np.linalg.norm(pts, axis=1)
        margin = 1.0 - rn
        bmask = margin < r_est
        # push inward
        F[bmask] += (pts[bmask] / np.where(rn[bmask] > 1e-12, rn[bmask], 1.0)[:, None]) * (1.0 / (margin[bmask] + 1e-6))
        pts = pts + step * F
        # clamp inside
        rn = np.linalg.norm(pts, axis=1)
        over = rn > (1.0 - 1e-9)
        if np.any(over):
            pts[over] = pts[over] / rn[over] * (1.0 - 1e-9)
        s = _score(pts)
        if s > best_s:
            best_s = s
            best = pts.copy()

    # --- Main adaptive loop ---
    pts = best.copy()
    best_s = _score(pts)
    it = 0
    base_step = 1e-4
    while remaining() > 30.0:
        it += 1
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        r_est = max(best_s, 1e-6)
        dmin = 2.0 * r_est
        inv = np.zeros_like(d)
        mask = d < dmin
        inv[mask] = 1.0 / (d[mask] ** 2 + 1e-12)
        diff = pts[None, :, :] - pts[:, None, :]
        d_safe = np.where(d > 1e-12, d, 1.0)
        diff_n = diff / d_safe[:, :, None]
        F = (inv[:, :, None] * diff_n).sum(axis=1)
        rn = np.linalg.norm(pts, axis=1)
        margin = 1.0 - rn
        bmask = margin < r_est
        F[bmask] += (pts[bmask] / np.where(rn[bmask] > 1e-12, rn[bmask], 1.0)[:, None]) * (1.0 / (margin[bmask] + 1e-6))
        # adaptive step: smaller as we get better
        step = base_step / (1.0 + 0.02 * it)
        pts = pts + step * F
        rn = np.linalg.norm(pts, axis=1)
        over = rn > (1.0 - 1e-9)
        if np.any(over):
            pts[over] = pts[over] / rn[over] * (1.0 - 1e-9)
        s = _score(pts)
        if s > best_s:
            best_s = s
            best = pts.copy()

    # --- Final high-precision polish: tiny steps, more iterations ---
    if remaining() > 5.0:
        pts = best.copy()
        base_step2 = 1e-6
        for it2 in range(2000):
            if remaining() < 2.0:
                break
            d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            r_est = max(best_s, 1e-6)
            dmin = 2.0 * r_est
            inv = np.zeros_like(d)
            mask = d < dmin
            inv[mask] = 1.0 / (d[mask] ** 2 + 1e-12)
            diff = pts[None, :, :] - pts[:, None, :]
            d_safe = np.where(d > 1e-12, d, 1.0)
            diff_n = diff / d_safe[:, :, None]
            F = (inv[:, :, None] * diff_n).sum(axis=1)
            rn = np.linalg.norm(pts, axis=1)
            margin = 1.0 - rn
            bmask = margin < r_est
            F[bmask] += (pts[bmask] / np.where(rn[bmask] > 1e-12, rn[bmask], 1.0)[:, None]) * (1.0 / (margin[bmask] + 1e-6))
            step = base_step2 / (1.0 + 0.005 * it2)
            pts = pts + step * F
            rn = np.linalg.norm(pts, axis=1)
            over = rn > (1.0 - 1e-9)
            if np.any(over):
                pts[over] = pts[over] / rn[over] * (1.0 - 1e-9)
            s = _score(pts)
            if s > best_s:
                best_s = s
                best = pts.copy()

    # Final safety: ensure validity (clip any tiny boundary violations)
    rn = np.linalg.norm(best, axis=1)
    over = rn > (1.0 - 1e-12)
    if np.any(over):
        best[over] = best[over] / rn[over] * (1.0 - 1e-12)
    return best
