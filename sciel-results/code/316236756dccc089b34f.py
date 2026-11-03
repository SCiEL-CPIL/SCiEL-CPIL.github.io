"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 100


def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum for precision."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _fast_energy(pts):
    """Fast vectorized energy for comparison during optimization."""
    n = pts.shape[0]
    D = pts @ pts.T
    iu = np.triu_indices(n, 1)
    val = 1.0 - D[iu]
    if np.any(val < 1e-12):
        return float('inf')
    terms = -0.5 * (np.log(2.0) + np.log(val))
    return float(np.sum(terms))


def _gradient(pts):
    """Gradient of E = sum_{i<j} log(1/|x_i - x_j|).
    grad_i = sum_{j != i} (x_j - x_i)/|x_i - x_j|^2  (in ambient R^3)."""
    n = pts.shape[0]
    diff = pts[None, :, :] - pts[:, None, :]          # diff[i,j] = x_j - x_i
    dist2 = np.sum(diff * diff, axis=2)               # (n, n)
    np.fill_diagonal(dist2, 1e30)
    weights = 1.0 / dist2
    np.fill_diagonal(weights, 0.0)
    return np.sum(weights[:, :, None] * diff, axis=1)


def _tangent(g, x):
    """Project g onto tangent space of sphere at x."""
    return g - np.sum(g * x, axis=1, keepdims=True) * x


def _fibonacci(k):
    """Fibonacci-spiral initial configuration."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _line_search(x0, d, e0, max_iter=40):
    """Bisection line search along geodesic direction d (unit tangent)."""
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        x = x0 * math.cos(mid) + d * math.sin(mid)
        e = _fast_energy(x)
        if not np.isfinite(e):
            hi = mid
            continue
        if e < e0:
            lo = mid
        else:
            hi = mid
    x = x0 * math.cos(lo) + d * math.sin(lo)
    return _normalize(x)


def _refine(x0, deadline):
    """Refine x0 with steepest descent + line search until time budget."""
    x = x0.copy()
    e = _fast_energy(x)
    if not np.isfinite(e):
        return x0, float('inf')
    for _ in range(5000):
        if time.time() > deadline:
            break
        g = _tangent(_gradient(x), x)
        gn = np.linalg.norm(g)
        if gn < 1e-10:
            break
        d = g / gn
        x = _line_search(x, d, e)
        e = _fast_energy(x)
        if not np.isfinite(e):
            break
    return x, e


def construct():
    """Place 100 points on the unit sphere to minimize logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Multi-restart: Fibonacci spiral plus jittered variants to escape minima.
    rng = np.random.default_rng(12345)
    inits = [_fibonacci(0)]
    for k in range(1, 6):
        base = _fibonacci(0)
        jitter = rng.normal(0.0, 0.12, size=(N, 3))
        inits.append(_normalize(base + jitter))

    best_pts = inits[0]
    best_e = _fast_energy(best_pts)

    for init in inits:
        if time.time() > deadline - 20.0:
            break
        cand, e = _refine(init, deadline - 10.0)
        if np.isfinite(e) and e < best_e:
            best_e = e
            best_pts = cand

    final_energy = _log_energy(best_pts)
    if not np.isfinite(final_energy):
        return _fibonacci(0)
    return best_pts
# EVOLVE-BLOCK-END
