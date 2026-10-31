import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 16
D = 2

def _get_dists(x):
    """Calculate all pairwise distances."""
    # x is (N, D)
    diff = x[:, None, :] - x[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    # Get upper triangle indices to avoid duplicates and zeros
    i, j = np.triu_indices(N, k=1)
    return dists[i, j]

def _objective_smooth(x, alpha):
    """
    Smoothed objective function.
    x: (N, D) array
    alpha: smoothing parameter
    """
    dists = _get_dists(x)
    # Avoid log(0) if points coincide
    dists = np.maximum(dists, 1e-10)
    
    # Soft max
    log_sum_exp_max = np.log(np.sum(np.exp(alpha * dists)))
    d_max_smooth = log_sum_exp_max / alpha
    
    # Soft min
    log_sum_exp_min = np.log(np.sum(np.exp(-alpha * dists)))
    d_min_smooth = -log_sum_exp_min / alpha
    
    ratio = d_max_smooth / d_min_smooth
    return ratio**2

def _objective_raw(x):
    """
    Raw objective function for final evaluation or fallback.
    """
    dists = _get_dists(x)
    if np.min(dists) < 1e-10:
        return 1e10
    d_min = np.min(dists)
    d_max = np.max(dists)
    return (d_max / d_min)**2

def _hexagonal_init(n_points):
    """
    Generate an initial configuration based on a hexagonal lattice.
    """
    points = []
    # Hexagonal spacing
    dx = 1.0
    dy = np.sqrt(3) / 2.0
    
    # Generate a grid of hex points
    # We need to select 16 points that form a roughly circular shape
    # Let's generate a 5x5 hex grid and pick the closest 16 to the center
    candidates = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            # Hex coordinates to Cartesian
            x = i * dx + (j % 2) * dx / 2.0
            y = j * dy
            candidates.append((x, y))
    
    candidates = np.array(candidates)
    # Center
    center = candidates.mean(axis=0)
    candidates = candidates - center
    
    # Sort by distance from center
    dists_to_center = np.linalg.norm(candidates, axis=1)
    indices = np.argsort(dists_to_center)
    
    # Take the first 16
    return candidates[indices[:n_points]]

def _square_init(n_points):
    """
    Generate an initial configuration based on a square grid.
    """
    side = int(np.ceil(np.sqrt(n_points)))
    grid = list(itertools.product(range(side), repeat=2))[:n_points]
    points = np.array(grid, dtype=float)
    # Center
    points -= points.mean(axis=0)
    return points

def _random_init(n_points, scale=1.0):
    """
    Generate random initial configuration.
    """
    return np.random.randn(n_points, 2) * scale

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin from 120s limit
    
    best_score = float('inf')
    best_points = None
    
    # Initial configurations
    inits = [
        _hexagonal_init(N),
        _square_init(N),
        _random_init(N, 2.0),
        _random_init(N, 2.0),
        _random_init(N, 2.0)
    ]
    
    # Optimize each initial configuration
    for init in inits:
        if time.time() > deadline:
            break
            
        current_points = init.copy()
        
        # Annealing alpha for smooth objective
        alphas = [10, 50, 200]
        
        for alpha in alphas:
            if time.time() > deadline:
                break
                
            try:
                res = minimize(
                    _objective_smooth,
                    current_points,
                    args=(alpha,),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-10}
                )
                current_points = res.x
            except Exception:
                pass
        
        # Final polish with raw objective using Powell (derivative-free, robust to non-smoothness)
        # Only do this if we have time
        if time.time() < deadline - 5.0:
            try:
                res = minimize(
                    _objective_raw,
                    current_points,
                    method='Powell',
                    options={'maxiter': 500, 'ftol': 1e-8}
                )
                current_points = res.x
            except Exception:
                pass
        
        # Evaluate final score
        score = _objective_raw(current_points)
        
        if score < best_score:
            best_score = score
            best_points = current_points
    
    # If no valid points found (shouldn't happen), return grid
    if best_points is None:
        best_points = _square_init(N)
        
    return best_points
