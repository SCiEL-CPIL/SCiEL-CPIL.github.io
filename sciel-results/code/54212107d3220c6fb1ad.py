"""Optimized logarithmic energy minimization for 100 points on the sphere.

Strategy: multi-restart from diverse initializations (Fibonacci, random, jittered,
antipodal) combined with a proper Riemannian L-BFGS-B (retraction-based line search
with exact energy and Riemannian gradient), followed by a tight final polish.
Energy/gradient use the cosine form d^2 = 2 - 2*cos(theta) to avoid a sqrt in the
hot loop, which lets us afford more restarts and a tighter polish within budget.
"""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import scipy.optimize as so

_DEADLINE = 130.0  # seconds; safety margin before the 150 s kill

def _energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return 1e18
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _energy_fast(pts):
    """Vectorized energy: E = -0.5 * sum_{i!=j} log(d_ij), d^2 = 2 - 2 cos(theta)."""
    n = len(pts)
    cos = pts @ pts.T
    np.fill_diagonal(cos, 1.0)
    d_sq = 2.0 - 2.0 * cos
    iu = np.triu_indices(n, k=1)
    d2 = d_sq[iu]
    if np.any(d2 < 1e-18):
        return 1e18
    # sum of log(1/d) over pairs = 0.5 * sum over i!=j of log(1/sqrt(d2))
    return float(-0.5 * np.sum(np.log(d2)) * 0.5)  # = -0.25 * sum log(d2)

def _energy_grad(pts):
    """Energy and ambient gradient via cosine form (no sqrt).

    E = -0.5 * sum_{i!=j} log d_ij
    dE/dx_i = 0.5 * sum_{j!=i} (x_i - x_j) / d_ij^2
           = 0.25 * sum_{j!=i} (x_i - x_j) / (2 - 2 x_i.x_j)
    """
    n = len(pts)
    cos = pts @ pts.T
    np.fill_diagonal(cos, 1.0)
    d_sq = 2.0 - 2.0 * cos
    np.fill_diagonal(d_sq, 1.0)
    inv = 0.25 / d_sq
    np.fill_diagonal(inv, 0.0)
    grad = (pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) * inv[:, :, np.newaxis]
    grad = grad.sum(axis=1)

    iu = np.triu_indices(n, k=1)
    d2 = d_sq[iu]
    if np.any(d2 < 1e-18):
        return 1e18, np.zeros_like(pts)
    E = float(-0.25 * np.sum(np.log(d2)))
    return E, grad

def _tangent_project(pts, vec):
    """Project vector onto tangent space at each point."""
    dot = np.sum(vec * pts, axis=1, keepdims=True)
    return vec - dot * pts

def _retract(pts, v, alpha=1.0):
    """Retraction on the sphere: normalize pts + alpha*v."""
    new_pts = pts + alpha * v
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return new_pts / norms

def _project(pts):
    """Project onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_start():
    """Golden-angle Fibonacci spiral on the sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project(pts)

def _random_start(seed):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return _project(pts)

def _jittered_fibonacci(jitter, seed):
    """Fibonacci with random tangent jitter."""
    pts = _fibonacci_start()
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((N, 3)) * jitter
    noise = _tangent_project(pts, noise)
    return _retract(pts, noise, 1.0)

def _antipodal_start():
    """Fibonacci start with a random antipodal flip of half the points."""
    pts = _fibonacci_start()
    rng = np.random.default_rng(99)
    flip = (rng.random(N) < 0.5).astype(float) * 2.0 - 1.0
    pts = pts * flip[:, np.newaxis]
    return _project(pts)

def _lbfgs_optimize(pts, max_iter=2000, tol=1e-12):
    """Riemannian L-BFGS-B: minimizes in ambient coords but the line search
    retracts onto the sphere, so the iterates always stay on the manifold and the
    objective is the true on-sphere energy. Tangent-projected gradient guides the
    L-BFGS-B direction; the retraction handles the constraint."""
    pts = _project(pts.copy())

    def objective(x_flat):
        x = x_flat.reshape(N, 3)
        return _energy_fast(x)

    def gradient(x_flat):
        x = x_flat.reshape(N, 3)
        _, grad = _energy_grad(x)
        return _tangent_project(x, grad).flatten()

    x0 = pts.flatten()
    try:
        result = so.minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                             options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol,
                                      'maxls': 40, 'maxfun': max_iter * 4})
        opt_pts = _project(result.x.reshape(N, 3))
        opt_E = _energy_fast(opt_pts)
        return opt_pts, opt_E
    except Exception:
        return pts, _energy_fast(pts)

def _local_polish(pts, max_iter=8000, tol=1e-14):
    """High-precision local polish with very tight tolerance."""
    return _lbfgs_optimize(pts, max_iter=max_iter, tol=tol)

def construct():
    start_time = time.time()

    best_pts = None
    best_E = 1e18

    # ---- Build a diverse set of initializations (cheap) ----
    initializations = []
    initializations.append(_fibonacci_start())
    initializations.append(_antipodal_start())
    for seed in [42, 123, 456, 789, 1000, 2024, 31415, 27182, 16180, 82718,
                 14142, 777, 555, 333, 111, 2025, 602, 9999, 12345, 54321]:
        initializations.append(_random_start(seed))
    for jitter, seed in [(0.01, 100), (0.03, 101), (0.05, 102), (0.08, 103),
                         (0.12, 104), (0.18, 105), (0.25, 106)]:
        initializations.append(_jittered_fibonacci(jitter, seed))

    # ---- Phase 1: multi-start Riemannian L-BFGS with adaptive time budget ----
    for i, init_pts in enumerate(initializations):
        if time.time() > _DEADLINE - 25:
            break
        elapsed = time.time() - start_time
        remaining = _DEADLINE - 25 - elapsed
        n_remaining = len(initializations) - i
        # Per-start budget: enough for one solid L-BFGS run, capped.
        budget = min(remaining / max(1, n_remaining), 8.0)
        if budget < 2.0:
            break

        # Coarse run
        opt_pts, E = _lbfgs_optimize(init_pts, max_iter=1500, tol=1e-11)
        if E < best_E:
            best_E = E
            best_pts = opt_pts

        # If this start is clearly the best so far, give it a quick extra polish.
        if E < best_E and time.time() < _DEADLINE - 20:
            polished, pE = _lbfgs_optimize(best_pts, max_iter=1500, tol=1e-13)
            if pE < best_E:
                best_E = pE
                best_pts = polished

    # ---- Phase 2: tighten the incumbent with a few more high-precision passes ----
    if best_pts is not None:
        for _ in range(3):
            if time.time() > _DEADLINE - 8:
                break
            polished, pE = _local_polish(best_pts, max_iter=4000, tol=1e-14)
            if pE < best_E:
                best_E = pE
                best_pts = polished
            else:
                break

    # ---- Fallback ----
    if best_pts is None:
        best_pts = _fibonacci_start()

    return _project(best_pts)
# EVOLVE-BLOCK-END
