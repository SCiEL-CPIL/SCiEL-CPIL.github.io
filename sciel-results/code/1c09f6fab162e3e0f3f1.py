"""Pack 50 equal circles in a unit square via smooth optimization + local polish."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 115.0  # safety margin before 120s limit
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _score(x):
    """Compute the max radius for a given set of centers in [-0.5, 0.5]^2.
    x is shape (N, 2)."""
    c = x.reshape(-1, 2)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    # Distance to walls
    wall = 0.5 - np.abs(c)
    min_wall = np.min(wall)
    r = min(min_pair / 2.0, min_wall)
    return r


def _neg_score(x):
    return -_score(x)


def _soft_min(a, beta):
    """Soft minimum of array a using log-sum-exp trick for stability."""
    a = np.asarray(a)
    m = np.min(a)
    # Avoid overflow: softmin ≈ m + log(sum(exp(-(a-m)/beta))) / beta
    # Actually softmin(a) ≈ -logsumexp(-a/beta)/beta
    # Let's use: -log(sum(exp(-a/beta)))/beta
    # For numerical stability:
    # = -log(exp(-m/beta) * sum(exp(-(a-m)/beta))) / beta
    # = m/beta + log(sum(exp(-(a-m)/beta))) / beta
    # Wait, let me redo:
    # softmin(a, beta) = -1/beta * log(sum_i exp(-a_i/beta))
    # = -1/beta * log(exp(-m/beta) * sum_i exp(-(a_i-m)/beta))
    # = -1/beta * (-m/beta + log(sum_i exp(-(a_i-m)/beta)))
    # = m/beta - 1/beta * log(sum_i exp(-(a_i-m)/beta))
    # Hmm, that doesn't look right either. Let me just use the standard formula.
    # softmin(a) = -logsumexp(-a)/beta where logsumexp is natural log.
    # Actually: softmin(a, beta) = -1/beta * log(sum(exp(-a_i/beta)))
    # Let s = sum(exp(-(a_i - m)/beta)) where m = min(a)
    # Then sum(exp(-a_i/beta)) = exp(-m/beta) * s
    # softmin = -1/beta * log(exp(-m/beta) * s) = -1/beta * (-m/beta + log(s)) = m/beta - log(s)/beta
    # That gives softmin ≈ m when beta is large (s ≈ N, log(N)/beta → 0)
    # But we want softmin ≈ m, and m/beta - log(s)/beta... if beta is large, m/beta is small.
    # I think I'm confusing myself. Let me just use a simpler approach.
    
    # Simple softmin: m - (1/beta) * log(sum(exp(-(a-m)/beta)))
    # When beta→∞, this → m - log(N)/beta → m. Good.
    # When beta→0+, this → m. Good.
    shifted = -(a - m) / beta
    # Clip for numerical stability
    shifted = np.clip(shifted, -50, 50)
    s = np.sum(np.exp(shifted))
    return m - np.log(s) / beta


def _objective(x, beta):
    """Smooth objective: maximize min(pairwise_dist/2, wall_dist).
    Using soft-min to make it differentiable."""
    c = x.reshape(-1, 2)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    # Exclude diagonal
    mask = ~np.eye(N, dtype=bool)
    pair_dists = dist[mask]
    wall = 0.5 - np.abs(c)
    wall_dists = wall.flatten()
    
    # Combine all constraints into one array
    all_dists = np.concatenate([pair_dists, wall_dists])
    # We want to maximize the minimum of (pair_dists/2, wall_dists)
    # So define values = [pair_dists/2, wall_dists] and softmin them
    values = np.concatenate([pair_dists / 2.0, wall_dists])
    r_soft = _soft_min(values, beta)
    return -r_soft


def _construct():
    global _start_time
    _start_time = time.time()
    
    best_x = None
    best_r = -np.inf
    
    # Helper to update best
    def _update_best(x):
        nonlocal best_x, best_r
        r = _score(x)
        if r > best_r:
            best_r = r
            best_x = x.copy()
    
    # Phase 1: Differential Evolution for global search
    # Initial bounds: centers in [-0.49, 0.49] to stay safely inside
    bounds = [(-0.49, 0.49)] * (2 * N)
    
    if _elapsed() < DEADLINE - 60:
        try:
            de_result = differential_evolution(
                _neg_score, bounds, seed=42, maxiter=50, popsize=15,
                tol=1e-6, polish=False, init='sobol'
            )
            _update_best(de_result.x)
        except Exception:
            pass
    
    # Phase 2: L-BFGS local refinement with smooth surrogate
    if best_x is not None and _elapsed() < DEADLINE - 30:
        # Try multiple beta values for smooth optimization
        for beta in [1000, 500, 200, 100, 50, 20, 10, 5, 2, 1]:
            if _elapsed() >= DEADLINE - 15:
                break
            try:
                res = minimize(
                    lambda x: _objective(x, beta),
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 10000}
                )
                _update_best(res.x)
            except Exception:
                pass
    
    # Phase 3: High-precision polish with very smooth surrogate
    if best_x is not None and _elapsed() < DEADLINE - 10:
        for beta in [10000, 5000, 2000, 1000, 500, 200, 100, 50, 20, 10, 5, 2, 1, 0.5]:
            if _elapsed() >= DEADLINE - 5:
                break
            try:
                res = minimize(
                    lambda x: _objective(x, beta),
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 20000}
                )
                _update_best(res.x)
            except Exception:
                pass
    
    # Phase 4: Final direct optimization on the exact (nonsmooth) objective
    if best_x is not None and _elapsed() < DEADLINE - 5:
        try:
            res = minimize(
                _neg_score,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 50000}
            )
            _update_best(res.x)
        except Exception:
            pass
    
    # Final validation and clamping
    if best_x is not None:
        c = best_x.reshape(-1, 2)
        # Ensure centers are strictly inside the square
        c = np.clip(c, -0.499999, 0.499999)
        best_x = c.flatten()
    
    # Fallback: grid if nothing worked
    if best_x is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_x = grid[:N].flatten()
    
    return best_x.reshape(N, 2)


def construct():
    """Construct centers of 50 equal circles in [-0.5, 0.5]^2."""
    result = _construct()
    # Final sanity check
    if not np.all(np.isfinite(result)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        result = grid[:N]
    return result
