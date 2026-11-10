# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial.distance import pdist

_T0 = time.time()
_DEADLINE = _T0 + 540.0  # safety margin well under the 630s wall-clock kill


def _fibonacci_sphere(n):
    """Evenly spread n points on the unit sphere (Fibonacci lattice)."""
    i = np.arange(n, dtype=float)
    phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))


def _softmin_logsumexp(vals, beta):
    """log-sum-exp of vals (robust)."""
    m = np.max(vals)
    return m + np.log(np.sum(np.exp(vals - m)))


def _softmax_logsumexp(vals, beta):
    """log-sum-exp of vals (robust)."""
    m = np.max(vals)
    return m + np.log(np.sum(np.exp(vals - m)))


def _softmin(vals, beta):
    """Smooth approximation of min(vals)."""
    return -_softmin_logsumexp(-vals / beta, beta)


def _softmax(vals, beta):
    """Smooth approximation of max(vals)."""
    return _softmax_logsumexp(vals / beta, beta)


def _score(pts):
    """(d_min/d_max)^2 for a point set; 0.0 if degenerate."""
    d = pdist(pts)
    if d.size == 0:
        return 0.0
    dmin = float(np.min(d))
    dmax = float(np.max(d))
    if dmax <= 1e-12:
        return 0.0
    r = dmin / dmax
    return r * r


def _repel_step(pts, beta, lam, lr):
    """One gradient step on log(softmin(d)) - log(softmax(d))."""
    n = pts.shape[0]
    P = pts
    diff = P[:, None, :] - P[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2) + 1e-30)
    # zero the diagonal for safety
    np.fill_diagonal(d, 0.0)
    d_safe = np.maximum(d, 1e-30)

    # log(softmin(d)) = -logsumexp(-d/beta)
    a = -d_safe / beta
    amax = np.max(a)
    ea = np.exp(a - amax)
    ea[np.diag_indices_from(ea)] = 0.0
    smin = amax + np.log(np.sum(ea))          # logsumexp(-d/beta)

    # log(softmax(d)) = logsumexp(d/beta)
    b = d_safe / beta
    bmax = np.max(b)
    eb = np.exp(b - bmax)
    eb[np.diag_indices_from(eb)] = 0.0
    smax = bmax + np.log(np.sum(eb))          # logsumexp(d/beta)

    # d/dP_i  log(softmin(d))
    wmin = ea / np.sum(ea)                    # weights on -d/beta
    gmin = (1.0 / beta) * np.sum(diff * wmin[:, :, None], axis=1)

    # d/dP_i  log(softmax(d))
    wmax = eb / np.sum(eb)                    # weights on d/beta
    gmax = (1.0 / beta) * np.sum(diff * wmax[:, :, None], axis=1)

    grad = gmin - gmax
    step = lr * grad
    # clip step magnitude to keep the trajectory stable
    step_norm = np.linalg.norm(step, axis=1, keepdims=True)
    max_step = 0.25
    scale = np.where(step_norm > max_step, max_step / (step_norm + 1e-30), 1.0)
    step = step * scale
    new = P + step
    # gentle centering to avoid drift
    new = new - np.mean(new, axis=0, keepdims=True)
    return new


def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3

    # Best-so-far tracking
    best_pts = _fibonacci_sphere(n)
    best_score = _score(best_pts)

    # Coarse-to-fine repel schedule (higher beta = smoother / coarser)
    betas = [0.6, 0.35, 0.2, 0.12, 0.07, 0.04, 0.025, 0.015, 0.01, 0.007, 0.005]
    lam = 0.03

    rng = np.random.default_rng(12345)

    for beta in betas:
        # adaptive learning rate: shrink as beta shrinks for stability
        lr = min(0.05, 0.02 * beta + 0.002)
        iters = 3000
        for it in range(iters):
            if time.time() > _DEADLINE:
                break
            pts = _repel_step(best_pts, beta, lam, lr)
            if not np.all(np.isfinite(pts)):
                break
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        if time.time() > _DEADLINE:
            break

    # A few randomized restarts at a small beta if we have budget left
    if time.time() < _DEADLINE - 60.0:
        for _ in range(4):
            if time.time() > _DEADLINE - 60.0:
                break
            start = _fibonacci_sphere(n) + 0.15 * rng.standard_normal((n, d))
            for beta in [0.2, 0.1, 0.05, 0.02, 0.01]:
                if time.time() > _DEADLINE - 60.0:
                    break
                lr = min(0.05, 0.02 * beta + 0.002)
                for it in range(1500):
                    if time.time() > _DEADLINE - 60.0:
                        break
                    pts = _repel_step(start, beta, lam, lr)
                    if not np.all(np.isfinite(pts)):
                        break
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_pts = pts.copy()

    # Final validation: guarantee finite, correct-shape output
    out = np.asarray(best_pts, dtype=float)
    out = np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
    if out.shape != (n, d):
        out = _fibonacci_sphere(n)
    return out


# EVOLVE-BLOCK-END
