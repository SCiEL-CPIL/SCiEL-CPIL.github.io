"""Search-based packer for 100 equal spheres in a unit cube."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 100

def _score(points):
    """Return the maximal radius for the given set of centres."""
    if len(points) != N:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Inside the cube [-0.5, 0.5]^3
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r_pair = min_dist / 2.0
    return min(r_pair, r_margin)

def _soft_min(a, b, beta=50.0):
    """Smooth approximation of min(a, b) using log-sum-exp."""
    m = np.minimum(a, b)
    return m - np.logaddexp(-beta * (a - m), -beta * (b - m)) / beta

def _neg_soft_radius(x, beta=50.0):
    """Smooth objective: -soft_min(r_pair, r_margin)."""
    pts = x.reshape(N, 3)
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return 1e10
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 1e10
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = dists.min() / 2.0
    return -_soft_min(r_pair, r_margin, beta)

def _random_grid_initial():
    """Generate a simple grid-based initialisation."""
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.5 + 0.1, 0.5 - 0.1, nx)
    ys = np.linspace(-0.5 + 0.1, 0.5 - 0.1, ny)
    zs = np.linspace(-0.5 + 0.125, 0.5 - 0.125, nz)
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return points

def _local_optimize(points, method='L-BFGS-B', beta=50.0, maxiter=80):
    """Locally optimise using smooth soft-min surrogate."""
    x0 = points.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def neg_radius(x):
        return _neg_soft_radius(x, beta)
    
    try:
        result = minimize(neg_radius, x0, method=method, bounds=bounds,
                          options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-7})
        if result.fun < neg_radius(x0):
            return result.x.reshape(N, 3)
        else:
            return points
    except Exception:
        return points

def _high_precision_polish(points, maxiter=200):
    """Final polish with tighter tolerances and higher beta for sharper min."""
    # Use progressively higher beta to sharpen the soft-min
    for beta in [100.0, 200.0, 500.0]:
        x0 = points.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        def neg_radius(x):
            return _neg_soft_radius(x, beta)
        
        try:
            result = minimize(neg_radius, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
            if result.fun < neg_radius(x0):
                points = result.x.reshape(N, 3)
        except Exception:
            pass
    return points

def _simulated_annealing(points, t_max=5.0, t_min=0.01, alpha=0.99, max_iter=500):
    """Simple Simulated Annealing to escape local minima."""
    current_points = points.copy()
    current_score = _score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    t = t_max
    for i in range(max_iter):
        if t < t_min:
            break
        idx = random.randint(0, N - 1)
        move_size = 0.05 * t
        delta = np.random.normal(0, move_size, 3)
        new_points = current_points.copy()
        new_points[idx] += delta
        new_points[idx] = np.clip(new_points[idx], -0.5, 0.5)
        
        new_score = _score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_points = current_points.copy()
                best_score = current_score
        else:
            delta_e = current_score - new_score
            if delta_e < 0:
                accept = True
            else:
                accept = random.random() < np.exp(-delta_e / t)
            
            if accept:
                current_points = new_points
                current_score = new_score
        
        t *= alpha
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial solution from grid
    points = _random_grid_initial()
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Phase 1: Local optimisation with soft-min
    if time.time() < deadline - 30:
        opt_points = _local_optimize(points, method='L-BFGS-B', beta=50.0, maxiter=80)
        opt_score = _score(opt_points)
        if opt_score > best_score:
            best_points = opt_points
            best_score = opt_score
    
    # Phase 2: Multi-start with random perturbations and local optimisation
    random.seed(42)
    
    while time.time() < deadline - 15:
        if random.random() < 0.5:
            new_start = best_points + np.random.normal(0, 0.05, (N, 3))
            new_start = np.clip(new_start, -0.5, 0.5)
        else:
            new_start = _random_grid_initial() + np.random.normal(0, 0.02, (N, 3))
            new_start = np.clip(new_start, -0.5, 0.5)
        
        opt_points = _local_optimize(new_start, method='L-BFGS-B', beta=50.0, maxiter=60)
        opt_score = _score(opt_points)
        
        if opt_score > best_score:
            best_points = opt_points
            best_score = opt_score
        
        if random.random() < 0.3 and time.time() < deadline - 25:
            sa_points = _simulated_annealing(best_points, max_iter=100)
            sa_score = _score(sa_points)
            if sa_score > best_score:
                best_points = sa_points
                best_score = sa_score
    
    # Phase 3: High-precision polish with increasing beta
    if time.time() < deadline - 5:
        polished = _high_precision_polish(best_points, maxiter=200)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
    
    return best_points

# EVOLVE-BLOCK-END
