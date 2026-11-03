"""Improved program for packing 50 equal circles in a unit square."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
# Best known packing radius (Boll, Donovan, Graham, Lubachevsky 2000)
KNOWN_R = 0.071377103865

def _radius(pts):
    """Compute the packing radius for given points in [-0.5, 0.5]^2."""
    pts = np.asarray(pts)
    n = pts.shape[0]
    if n < 2:
        return 0.0
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Distance to walls
    wall_dists = np.min(np.stack([
        pts[:, 0] + 0.5,   # left
        0.5 - pts[:, 0],   # right
        pts[:, 1] + 0.5,   # bottom
        0.5 - pts[:, 1]    # top
    ], axis=-1), axis=-1)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall) / 2.0


def _soft_min_radius_neg(pts, alpha=100.0):
    """
    Smooth surrogate for -min_pairwise_dist / 2.
    Uses a soft-min approximation to make it differentiable.
    """
    pts = np.asarray(pts)
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Soft min: -log-sum-exp(-dists * alpha) / alpha
    # But we need to be careful with large alpha
    # Use a simpler approach: min of distances approximated by softmin
    # Actually, let's use the fact that we want to maximize the minimum distance
    # So we minimize -min(dists)
    # Soft min of dists: exp(-alpha * min) ~ sum(exp(-alpha * d))
    # log(sum(exp(-alpha*d))) / alpha ≈ min(d) for large alpha
    # So -min(d) ≈ -log(sum(exp(-alpha*d))) / alpha
    
    # For numerical stability:
    min_d = np.min(dists)
    shifted = -alpha * (dists - min_d)
    # Avoid overflow
    shifted = np.where(np.isfinite(shifted), shifted, 0)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    soft_min = min_d - log_sum_exp / alpha
    return -soft_min / 2.0


def _wall_penalty(pts, alpha=100.0):
    """Penalty for points outside the square."""
    pts = np.asarray(pts)
    # Distance to each wall (negative if outside)
    dl = pts[:, 0] + 0.5
    dr = 0.5 - pts[:, 0]
    db = pts[:, 1] + 0.5
    dt = 0.5 - pts[:, 1]
    
    # Soft min of all wall distances
    all_d = np.concatenate([dl, dr, db, dt])
    min_d = np.min(all_d)
    shifted = -alpha * (all_d - min_d)
    shifted = np.where(np.isfinite(shifted), shifted, 0)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    soft_min = min_d - log_sum_exp / alpha
    return -soft_min / 2.0


def _objective(pts_flat, alpha=50.0):
    """
    Objective to minimize: -min_radius using smooth surrogates.
    """
    pts = pts_flat.reshape(-1, 2)
    # Combine soft min of pairwise and walls
    pair_part = _soft_min_radius_neg(pts, alpha)
    wall_part = _wall_penalty(pts, alpha)
    # The actual radius is min(pair_min, wall_min) / 2
    # We want to maximize min(pair_min, wall_min)
    # So minimize -min(pair_min, wall_min)
    # Approximate: use a weighted combination or the soft min of both
    # Simpler: just use the pairwise part as primary, and add wall penalty
    return pair_part + 10.0 * np.maximum(0, -wall_part)


def _gradient(pts_flat, alpha=50.0):
    """
    Numerical gradient (central differences) for the objective.
    """
    eps = 1e-7
    n = len(pts_flat)
    grad = np.zeros(n)
    f0 = _objective(pts_flat, alpha)
    for i in range(n):
        pts_plus = pts_flat.copy()
        pts_plus[i] += eps
        pts_minus = pts_flat.copy()
        pts_minus[i] -= eps
        grad[i] = (_objective(pts_plus, alpha) - _objective(pts_minus, alpha)) / (2 * eps)
    return grad


def _project_inside(pts):
    """Project points into the square [-0.5, 0.5]^2."""
    pts = np.clip(pts, -0.5, 0.5)
    return pts


def _grid_init():
    """Create a grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _random_init():
    """Create a random initialization inside the square."""
    return np.random.uniform(-0.5, 0.5, size=(N, 2))


def _local_refine(pts, alpha=50.0, max_iter=200, tol=1e-10):
    """
    Local refinement using L-BFGS-B with smooth surrogate.
    """
    pts_flat = pts.flatten()
    
    def obj(x):
        return _objective(x, alpha)
    
    result = minimize(
        obj,
        pts_flat,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8},
        bounds=[(-0.5, 0.5)] * (2 * N)
    )
    
    best_pts = result.x.reshape(-1, 2)
    best_pts = _project_inside(best_pts)
    return best_pts


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5-second safety margin
    
    # Start with grid initialization
    best_pts = _grid_init()
    best_r = _radius(best_pts)
    
    # Try several random initializations
    n_random = 10
    for i in range(n_random):
        if time.time() > deadline - 5.0:
            break
        
        pts = _random_init()
        r = _radius(pts)
        if r > best_r:
            best_pts = pts.copy()
            best_r = r
    
    # Local refinement on the best initial point
    if time.time() < deadline - 10.0:
        refined = _local_refine(best_pts, alpha=50.0, max_iter=300)
        r = _radius(refined)
        if r > best_r:
            best_pts = refined
            best_r = r
    
    # Additional refinement with different alpha values
    for alpha in [30.0, 100.0, 200.0]:
        if time.time() > deadline - 5.0:
            break
        
        refined = _local_refine(best_pts, alpha=alpha, max_iter=200)
        r = _radius(refined)
        if r > best_r:
            best_pts = refined
            best_r = r
    
    # Final polish: try to push toward known optimum
    if time.time() < deadline - 3.0:
        # If we're close to the known optimum, do a final high-precision pass
        if best_r > 0.07:
            refined = _local_refine(best_pts, alpha=500.0, max_iter=500, tol=1e-12)
            r = _radius(refined)
            if r > best_r:
                best_pts = refined
                best_r = r
    
    # Ensure all points are strictly inside
    best_pts = np.clip(best_pts, -0.499999999, 0.499999999)
    
    return best_pts
