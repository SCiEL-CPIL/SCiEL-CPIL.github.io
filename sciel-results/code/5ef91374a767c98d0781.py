"""
Pack 50 equal circles in the unit circle.

Strategy:
  1. Generate multiple diverse initial configurations (hexagonal lattice,
     ring-based, random with rejection sampling).
  2. For each, run a smooth soft-min optimisation using a log-sum-exp based
     surrogate for the minimum clearance, with multi-stage beta annealing
     to escape local minima.
  3. Use L-BFGS-B from scipy with the smooth objective, then a final
     high-precision polish with tighter tolerances.
  4. Track elapsed time and return the best valid configuration before the
     deadline.
"""

import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 50
DEADLINE = 110.0  # seconds; safety margin under the 120 s / 150 s limits
_start = time.time()


def _elapsed():
    return time.time() - _start


def _remaining():
    return DEADLINE - _elapsed()


def _pairwise_distances(pts):
    """Return matrix of pairwise Euclidean distances, shape (N, N)."""
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _min_clearance(pts):
    """Exact minimum clearance: min of (pairwise_dist/2) and (1 - |p_i|)."""
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_wall = np.min(1.0 - np.linalg.norm(pts, axis=1))
    return min(min_pair, min_wall)


def _soft_min_clearance(pts, beta):
    """
    Smooth surrogate for the minimum clearance using log-sum-exp.
    
    We want to maximise min( min_{i<j} d_ij/2, min_i (1-|p_i|) ).
    Equivalently, minimise -min(...) = max( -d_ij/2, |p_i|-1 ).
    
    Using soft-max: -soft_min = soft_max of the negative terms.
    
    Terms: for each pair (i,j): -d_ij/2 ; for each i: |p_i| - 1.
    soft_max(x) = (1/beta) * log( sum exp(beta * x_k) )
    
    So objective = (1/beta) * log( sum_pairs exp(-beta * d_ij/2) + sum_i exp(beta*(|p_i|-1)) )
    """
    pts = np.asarray(pts)
    n = len(pts)
    
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Only upper triangle
    iu = np.triu_indices(n, 1)
    pair_terms = -d[iu] / 2.0  # shape (M,)
    
    # Wall terms
    wall_terms = np.linalg.norm(pts, axis=1) - 1.0  # shape (n,)
    
    all_terms = np.concatenate([pair_terms, wall_terms])
    
    # Stable log-sum-exp
    m = np.max(all_terms)
    if m > 1e12:
        return m  # overflow guard
    lse = m + np.log(np.sum(np.exp(beta * (all_terms - m))))
    return lse / beta


def _soft_min_grad(pts, beta):
    """
    Gradient of the soft-min clearance objective w.r.t. pts.
    
    objective = (1/beta) * log( sum_k exp(beta * x_k) )
    where x_k are the negative clearance terms.
    
    d obj / d p_i = (1/beta) * sum_k [ exp(beta*x_k)/S * beta * dx_k/dp_i ]
                  = sum_k [ w_k * dx_k/dp_i ]
    where w_k = exp(beta*x_k) / sum_j exp(beta*x_j)
    """
    pts = np.asarray(pts)
    n = len(pts)
    
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=-1))
    iu = np.triu_indices(n, 1)
    
    pair_terms = -d[iu] / 2.0
    wall_terms = np.linalg.norm(pts, axis=1) - 1.0
    all_terms = np.concatenate([pair_terms, wall_terms])
    
    m = np.max(all_terms)
    if m > 1e12:
        return np.zeros_like(pts)
    exp_terms = np.exp(beta * (all_terms - m))
    S = np.sum(exp_terms)
    if S < 1e-300:
        return np.zeros_like(pts)
    weights = exp_terms / S  # shape (M + n,)
    
    n_pairs = len(pair_terms)
    pair_w = weights[:n_pairs]
    wall_w = weights[n_pairs:]
    
    grad = np.zeros_like(pts)
    
    # Pair contributions: term_k = -d_ij/2, d term index k corresponds to (i,j)
    for k in range(n_pairs):
        i, j = iu[0][k], iu[1][k]
        w = pair_w[k]
        if w < 1e-15:
            continue
        dij = d[i, j]
        if dij < 1e-12:
            continue
        # d(-d_ij/2)/d p_i = -(p_i - p_j) / (2 * d_ij)
        # d(-d_ij/2)/d p_j = +(p_i - p_j) / (2 * d_ij)
        diff_vec = (pts[i] - pts[j]) / (2.0 * dij)
        grad[i] -= w * diff_vec
        grad[j] += w * diff_vec
    
    # Wall contributions: term_i = |p_i| - 1
    for i in range(n):
        w = wall_w[i]
        if w < 1e-15:
            continue
        r = np.linalg.norm(pts[i])
        if r < 1e-12:
            continue
        # d(|p_i|-1)/d p_i = p_i / |p_i|
        grad[i] += w * (pts[i] / r)
    
    return grad


def _is_valid(pts, min_radius=1e-6):
    """Check validity: finite, inside container, no overlaps."""
    if not np.all(np.isfinite(pts)):
        return False
    if pts.shape != (N, 2):
        return False
    if np.any(np.linalg.norm(pts, axis=1) > 1.0 + 1e-9):
        return False
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 2.0 * min_radius:
        return False
    return True


def _score(pts):
    """Compute the exact score (minimum clearance)."""
    if not _is_valid(pts):
        return 0.0
    return _min_clearance(pts)


def _hex_lattice_init():
    """Generate a hexagonal lattice configuration inside the unit disk."""
    best = None
    for spacing in np.linspace(0.22, 0.35, 30):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            # Pick the N closest to center for a compact arrangement
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind="stable")[:N]
            pts = points[idx]
            if best is None or _score(pts) > _score(best):
                best = pts.copy()
    if best is None:
        # Fallback: uniform random
        best = _random_init(42)
    return best


def _ring_init():
    """Ring-based initialization: concentric rings of circles."""
    # 1 center + rings of 6, 12, 18, 13 = 50
    rings = [0, 6, 12, 18, 13]
    pts = [np.array([0.0, 0.0])]
    for r_idx, count in enumerate(rings[1:]):
        radius = 0.25 * (r_idx + 1)
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        ring_pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        pts.append(ring_pts)
    pts = np.vstack(pts)
    return pts


def _random_init(seed):
    """Random initialization with rejection sampling for non-overlap."""
    rng = np.random.RandomState(seed)
    pts = []
    for _ in range(10000):
        if len(pts) >= N:
            break
        angle = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0, 0.9)
        p = np.array([r * np.cos(angle), r * np.sin(angle)])
        if len(pts) == 0:
            pts.append(p)
        else:
            arr = np.array(pts)
            d = np.linalg.norm(arr - p, axis=1)
            if np.min(d) > 0.2:
                pts.append(p)
    if len(pts) < N:
        # Pad with random points (will be fixed by optimizer)
        while len(pts) < N:
            angle = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0, 0.8)
            pts.append(np.array([r * np.cos(angle), r * np.sin(angle)]))
    return np.array(pts[:N])


def _optimize_from(pts0, max_time=25.0):
    """
    Run multi-stage beta annealing optimisation from initial config pts0.
    Returns the best configuration found.
    """
    best_pts = pts0.copy()
    best_score = _score(pts0)
    
    # Beta schedule: start low (smooth), increase for sharper min
    betas = [10.0, 50.0, 200.0, 1000.0, 5000.0, 20000.0]
    
    current = pts0.copy()
    for beta in betas:
        if _remaining() < 3.0:
            break
        
        def objective(x):
            return _soft_min_clearance(x, beta)
        
        def gradient(x):
            return _soft_min_grad(x, beta)
        
        try:
            res = minimize(
                objective,
                current,
                jac=gradient,
                method="L-BFGS-B",
                options={
                    "maxiter": 300,
                    "ftol": 1e-12,
                    "gtol": 1e-10,
                    "maxfun": 500,
                }
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                current = res.x
                s = _score(current)
                if s > best_score:
                    best_score = s
                    best_pts = current.copy()
        except Exception:
            pass
        
        if _remaining() < 3.0:
            break
    
    # Final high-precision polish with very high beta
    if _remaining() > 5.0:
        for beta in [50000.0, 100000.0]:
            if _remaining() < 2.0:
                break
            try:
                res = minimize(
                    lambda x: _soft_min_clearance(x, beta),
                    best_pts,
                    jac=lambda x: _soft_min_grad(x, beta),
                    method="L-BFGS-B",
                    options={
                        "maxiter": 500,
                        "ftol": 1e-15,
                        "gtol": 1e-12,
                        "maxfun": 1000,
                    }
                )
                if res.x is not None and np.all(np.isfinite(res.x)):
                    s = _score(res.x)
                    if s > best_score:
                        best_score = s
                        best_pts = res.x.copy()
            except Exception:
                pass
    
    return best_pts


def construct():
    """
    Main entry point: construct centres of 50 equal circles in the unit circle.
    Returns array of shape (50, 2).
    """
    global _start
    _start = time.time()
    
    # Generate diverse initial configurations
    inits = []
    
    # Hexagonal lattice
    if _remaining() > 5.0:
        inits.append(_hex_lattice_init())
    
    # Ring-based
    if _remaining() > 5.0:
        inits.append(_ring_init())
    
    # Several random inits with different seeds
    for seed in [1, 7, 42, 123, 999]:
        if _remaining() > 5.0:
            inits.append(_random_init(seed))
    
    best_pts = None
    best_score = 0.0
    
    for i, init_pts in enumerate(inits):
        if _remaining() < 8.0:
            break
        
        # Allocate more time to earlier (better) inits
        time_budget = min(_remaining() - 8.0, 30.0)
        if time_budget < 3.0:
            break
        
        result = _optimize_from(init_pts, max_time=time_budget)
        s = _score(result)
        
        if s > best_score:
            best_score = s
            best_pts = result.copy()
    
    # If we still have time, do additional random restarts
    if _remaining() > 10.0 and best_pts is not None:
        for seed in [555, 777, 1000, 2024]:
            if _remaining() < 5.0:
                break
            init_pts = _random_init(seed)
            result = _optimize_from(init_pts, max_time=min(_remaining() - 5.0, 15.0))
            s = _score(result)
            if s > best_score:
                best_score = s
                best_pts = result.copy()
    
    if best_pts is None:
        # Absolute fallback
        best_pts = _hex_lattice_init()
    
    # Final validation and clipping
    if not np.all(np.isfinite(best_pts)):
        best_pts = _hex_lattice_init()
    
    # Ensure no point is outside the unit circle
    norms = np.linalg.norm(best_pts, axis=1)
    for i in range(N):
        if norms[i] > 1.0:
            best_pts[i] *= (1.0 - 1e-10) / norms[i]
    
    return best_pts
