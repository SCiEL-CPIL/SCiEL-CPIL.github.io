"""Packing 50 equal circles in a unit square [-1/2, 1/2]^2.
Uses epigraph formulation with multi-stage beta annealing for high-precision packing.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s kill
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _remaining():
    return max(0.0, DEADLINE - _elapsed())


def _min_dist(pts):
    """Compute the minimum distance between any pair of points and to the boundary."""
    n = len(pts)
    # Pairwise distances
    dx = np.subtract.outer(pts[:, 0], pts[:, 0])
    dy = np.subtract.outer(pts[:, 1], pts[:, 1])
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    # Distance to boundary
    dist_bound = np.minimum(0.5 - np.abs(pts), 0.5)  # min over x,y for each point
    min_bound = dist_bound.min()
    return min(min_pair, min_bound)


def _score(pts):
    """The radius is half the minimum distance to another circle or to the wall."""
    return _min_dist(pts) / 2.0


def _clamp(pts):
    """Ensure points stay inside [-0.5, 0.5]^2."""
    return np.clip(pts, -0.5, 0.5)


def _epigraph_objective(x, beta):
    """
    Objective function for epigraph optimization.
    x: flattened array of length 2*N + 1. First 2*N are coordinates, last is t.
    beta: sharpness parameter.
    """
    pts = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    # Compute distances to constraints
    # 1. Pairwise distances
    dx = np.subtract.outer(pts[:, 0], pts[:, 0])
    dy = np.subtract.outer(pts[:, 1], pts[:, 1])
    d_pair = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d_pair, np.inf)
    
    # 2. Distances to walls
    d_wall = np.minimum(0.5 - np.abs(pts), 0.5)
    d_wall_flat = d_wall.flatten()
    
    # Combine all distances
    d_all = np.concatenate([d_pair.flatten(), d_wall_flat])
    
    # Smooth min approximation: t + (1/beta) * log(sum(exp(-beta * d)))
    # To avoid overflow, use log-sum-exp trick
    # val = t + (1/beta) * logsumexp(-beta * d_all)
    # logsumexp(z) = max_z + log(sum(exp(z - max_z)))
    z = -beta * d_all
    max_z = np.max(z)
    if not np.isfinite(max_z):
        return 1e10
    log_sum_exp = max_z + np.log(np.sum(np.exp(z - max_z)))
    
    return t + (1.0 / beta) * log_sum_exp


def _optimize_epigraph(pts0, betas, max_iter_per_stage=2000):
    """
    Multi-stage optimization using epigraph formulation.
    """
    pts = pts0.copy()
    x0 = np.concatenate([pts.flatten(), [_min_dist(pts)]])
    
    current_x = x0
    
    for beta in betas:
        if _remaining() < 5.0:
            break
        
        def obj(x):
            return _epigraph_objective(x, beta)
        
        # Use L-BFGS-B for efficiency
        try:
            res = minimize(obj, current_x, method='L-BFGS-B', 
                           options={'maxiter': max_iter_per_stage, 'ftol': 1e-12})
            current_x = res.x
        except Exception:
            pass
        
        # Update current_x with the new best
        # Ensure t is not too small (it should approximate min_dist)
        # Actually, just keep the coordinates
        pts = current_x[:2*N].reshape(N, 2)
        # Clamp to be safe, though L-BFGS-B doesn't have bounds easily without extra setup
        # We rely on the penalty in the objective if we go out of bounds? 
        # No, the objective doesn't penalize out-of-bounds directly except via d_wall becoming negative.
        # If d_wall is negative, exp(-beta * negative) is huge, so it's a strong penalty.
        
    return pts


def _polish_nm(pts):
    """Polish using Nelder-Mead on the negative min-distance objective."""
    def objective(x):
        p = x.reshape(-1, 2)
        p = _clamp(p)
        return -_min_dist(p)

    x0 = pts.flatten()
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                            'adaptive': True})
    return res.x.reshape(-1, 2), -res.fun / 2.0


def _grid_init():
    """Create an initial grid-based packing."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _hex_init():
    """Create a hexagonal packing initialization."""
    pts = []
    # Hexagonal lattice
    dx = 1.0 / 7.0
    dy = dx * np.sqrt(3) / 2.0
    
    # Center the lattice
    # We need to fit 50 points.
    # Let's just generate a grid of hex points and pick the best 50.
    # Or simpler: use a grid and perturb slightly?
    # Let's stick to a standard grid but rotated or just use grid_init for now.
    # Actually, a simple grid is often a good start for 50.
    return _grid_init()


def _random_init(seed=0):
    """Random initialization inside the square."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 2))


def construct():
    global START_TIME
    START_TIME = time.time()

    best_pts = None
    best_score = 0.0

    # Define beta schedule: start smooth, end sharp
    betas = [10, 50, 200, 1000, 5000]
    
    # Initializations
    inits = [
        _grid_init(),
        _random_init(1),
        _random_init(2),
        _random_init(3)
    ]

    for init in inits:
        if _remaining() < 10.0:
            break
        
        # Run epigraph optimization
        opt_pts = _optimize_epigraph(init, betas, max_iter_per_stage=1000)
        
        # Polish with Nelder-Mead for final precision
        if _remaining() > 5.0:
            nm_pts, nm_score = _polish_nm(opt_pts)
            if nm_score > best_score:
                best_score = nm_score
                best_pts = nm_pts.copy()
        
        # Also check the unpolished result just in case
        if _score(opt_pts) > best_score:
            best_score = _score(opt_pts)
            best_pts = opt_pts.copy()

    # If no valid points found (shouldn't happen), fallback
    if best_pts is None:
        best_pts = _grid_init()

    # Final check: ensure all points are strictly inside
    best_pts = np.clip(best_pts, -0.4999999, 0.4999999)

    return best_pts
