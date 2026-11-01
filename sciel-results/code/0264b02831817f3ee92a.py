"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 140  # safety margin before 150s kill


def _pairwise_dist_sq(pts):
    """Compute squared pairwise distances for N x D points. Returns (N, N) matrix."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sum(diff * diff, axis=2)


def _objective_smooth(pts_flat, tau=0.05):
    """
    Smooth surrogate: soft-max / soft-min of squared distances.
    Uses log-sum-exp for numerical stability.
    """
    pts = pts_flat.reshape(N, D)
    dsq = _pairwise_dist_sq(pts)
    # Upper triangle indices
    iu = np.triu_indices(N, k=1)
    vals = dsq[iu]
    
    if np.any(vals < 1e-14):
        return 1e10
    
    # Soft max: tau * log(sum(exp(v/tau)))
    v_max = np.max(vals)
    log_sum_exp_max = v_max + tau * np.log(np.sum(np.exp((vals - v_max) / tau)))
    
    # Soft min: -tau * log(sum(exp(-v/tau)))
    v_min = np.min(vals)
    log_sum_exp_min = -v_min - tau * np.log(np.sum(np.exp(-(vals - v_min) / tau)))
    
    ratio = log_sum_exp_max / log_sum_exp_min
    return ratio


def _true_score(pts):
    """Compute the actual squared ratio: (max_d / min_d)^2 = max_dsq / min_dsq."""
    dsq = _pairwise_dist_sq(pts)
    iu = np.triu_indices(N, k=1)
    vals = dsq[iu]
    min_dsq = np.min(vals)
    max_dsq = np.max(vals)
    if min_dsq < 1e-14:
        return 1e10
    return max_dsq / min_dsq


def _random_init(rng):
    """Random initialization in unit hypercube, then center and scale."""
    pts = rng.uniform(-1, 1, size=(N, D))
    # Center
    pts -= pts.mean(axis=0)
    # Scale so that max pairwise distance is ~1
    dsq = _pairwise_dist_sq(pts)
    iu = np.triu_indices(N, k=1)
    max_dsq = np.max(dsq[iu])
    if max_dsq > 0:
        pts *= 1.0 / np.sqrt(max_dsq)
    return pts


def _local_optimize(pts, tau=0.05, max_iter=500, tol=1e-10):
    """Local optimization using L-BFGS-B on the smooth surrogate."""
    x0 = pts.flatten()
    res = minimize(
        _objective_smooth,
        x0,
        args=(tau,),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8}
    )
    return res.x.reshape(N, D)


def _polish(pts, max_iter=2000, tol=1e-12):
    """High-precision polish with decreasing tau."""
    current = pts.copy()
    for tau in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
        if time.time() > _DEADLINE:
            break
        current = _local_optimize(current, tau=tau, max_iter=max_iter, tol=tol)
    return current


def construct():
    rng = np.random.default_rng(42)
    
    # Start with a few structured initializations
    candidates = []
    
    # 1. Random restarts
    n_restarts = 50
    for i in range(n_restarts):
        if time.time() > _DEADLINE:
            break
        pts = _random_init(rng)
        # Quick local opt
        pts = _local_optimize(pts, tau=0.05, max_iter=200, tol=1e-8)
        candidates.append(pts)
    
    # 2. Add a few structured starting points (simplex-like, grid-based)
    # Simplex in 3D (4 points) + more
    for trial in range(5):
        if time.time() > _DEADLINE:
            break
        # Start from a regular tetrahedron and add random points
        tetra = np.array([
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ], dtype=float)
        tetra /= np.sqrt(3)
        # Add N-4 random points near the tetrahedron vertices
        extra = rng.uniform(-0.5, 0.5, size=(N - 4, D))
        pts = np.vstack([tetra, extra])
        pts -= pts.mean(axis=0)
        pts = _local_optimize(pts, tau=0.05, max_iter=200, tol=1e-8)
        candidates.append(pts)
    
    if not candidates:
        # Fallback: return a simple grid
        side = int(np.ceil(N ** (1 / D)))
        grid = np.array([[i, j, k] for i in range(side) for j in range(side) for k in range(side)][:N], dtype=float)
        return grid
    
    # Evaluate all candidates and keep the best
    best_pts = None
    best_score = np.inf
    for pts in candidates:
        if time.time() > _DEADLINE:
            break
        s = _true_score(pts)
        if s < best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Polish the best candidate
    if best_pts is not None:
        best_pts = _polish(best_pts)
        best_score = _true_score(best_pts)
    
    # Try a few more restarts if time permits, targeting the best region
    remaining_restarts = 20
    for i in range(remaining_restarts):
        if time.time() > _DEADLINE:
            break
        # Perturb the best solution
        noise = rng.normal(0, 0.1, size=(N, D))
        pts = best_pts + noise
        pts -= pts.mean(axis=0)
        pts = _local_optimize(pts, tau=0.05, max_iter=200, tol=1e-8)
        s = _true_score(pts)
        if s < best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Final polish
    if best_pts is not None:
        best_pts = _polish(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
