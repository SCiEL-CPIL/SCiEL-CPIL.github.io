"""Packing 50 equal circles in a unit square: two-phase radius maximization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 150.0  # wall-clock seconds; must finish well before evaluator kill

# Precompute pair indices for vectorized distance computation
_i_idx, _j_idx = np.triu_indices(N, k=1)
_PAIRS = np.stack([_i_idx, _j_idx], axis=1)  # shape (M, 2), M = N*(N-1)/2 = 1225


def _compute_radius(pts):
    """Compute the feasible radius for a set of centre points.
    Returns min(half_min_pair_dist, min_wall_margin)."""
    # Pairwise distances
    diffs = pts[_i_idx] - pts[_j_idx]
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    min_pair = np.min(dists) / 2.0

    # Wall margins
    abs_pts = np.abs(pts)
    wall_margin = np.min(0.5 - np.max(abs_pts, axis=1))

    return min(min_pair, wall_margin)


def _objective(pts):
    """Objective for SLSQP: maximize min distance -> minimize negative."""
    # Flatten
    x = pts.reshape(-1)
    # Pairwise distances
    diffs = x[_i_idx * 2] - x[_j_idx * 2]  # not quite right for 2D
    # Let's do it properly
    p = x.reshape(N, 2)
    diffs = p[_i_idx] - p[_j_idx]
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    min_pair = np.min(dists)
    # Wall constraints handled by bounds
    return -min_pair


def _feasibility_check(pts, r):
    """Check if all circles of radius r are non-overlapping and inside square."""
    # Pairwise: all distances >= 2r
    diffs = pts[_i_idx] - pts[_j_idx]
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    if np.any(dists < 2.0 * r - 1e-12):
        return False
    # Wall: all points within [-0.5+r, 0.5-r]
    if np.any(pts < -0.5 + r - 1e-12) or np.any(pts > 0.5 - r + 1e-12):
        return False
    return True


def _local_refine(pts, r_init, time_left):
    """Use SLSQP to maximize the minimum distance from initial point."""
    x0 = pts.reshape(-1)
    bounds = [(-0.5, 0.5)] * (2 * N)

    def obj(x):
        p = x.reshape(N, 2)
        diffs = p[_i_idx] - p[_j_idx]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        return -np.min(dists)

    def obj_grad(x):
        p = x.reshape(N, 2)
        diffs = p[_i_idx] - p[_j_idx]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        min_idx = np.argmin(dists)
        grad = np.zeros(2 * N)
        d = diffs[min_idx]
        dist = dists[min_idx]
        if dist > 1e-15:
            # d(-dist)/dx for the two points in the min pair
            unit = d / dist
            # grad of -dist w.r.t. p[i] = -unit, w.r.t. p[j] = +unit
            grad[_i_idx[min_idx] * 2] = -unit[0]
            grad[_i_idx[min_idx] * 2 + 1] = -unit[1]
            grad[_j_idx[min_idx] * 2] = unit[0]
            grad[_j_idx[min_idx] * 2 + 1] = unit[1]
        return grad

    result = minimize(obj, x0, method='SLSQP', bounds=bounds,
                      jac=obj_grad, options={'maxiter': 200, 'ftol': 1e-12})
    if result.success or result.fun < obj(x0):
        best_pts = result.x.reshape(N, 2)
        return best_pts
    return pts


def _bisection_radius(pts, r_lo, r_hi, time_left):
    """Bisect to find the maximum feasible radius for given arrangement topology."""
    # Actually for fixed topology (which pairs are closest), we can solve exactly.
    # But simpler: just compute the radius directly.
    return _compute_radius(pts)


def _random_init(seed):
    """Generate random initial positions inside the square."""
    rng = np.random.default_rng(seed)
    # Start with a rough grid perturbation
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    # Add random perturbation
    perturbation = rng.uniform(-0.02, 0.02, size=(N, 2))
    pts = grid + perturbation
    # Clip to valid region
    pts = np.clip(pts, -0.45, 0.45)
    return pts


def _grid_init():
    """Standard grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    return grid


def construct():
    start_time = time.time()
    
    # Start with grid
    best_pts = _grid_init()
    best_r = _compute_radius(best_pts)
    
    # Try several random restarts with local refinement
    rng = np.random.default_rng(42)
    
    for trial in range(200):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:  # safety margin
            break
        
        # Generate diverse initial points
        seed = rng.integers(0, 2**31)
        pts = _random_init(seed)
        
        # Local refinement
        pts = _local_refine(pts, best_r, DEADLINE - elapsed)
        
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # Final refinement on best solution
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        best_pts = _local_refine(best_pts, best_r, DEADLINE - elapsed - 2)
        best_r = _compute_radius(best_pts)
    
    # Ensure validity: clip to square boundary
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    # Final safety check
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 2:
        # One more round of refinement
        best_pts = _local_refine(best_pts, best_r, DEADLINE - elapsed - 1)
        best_r = _compute_radius(best_pts)
        best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
