"""Rewritten program for packing 200 equal circles in a unit square."""
import numpy as np
from scipy.optimize import minimize
import time
import random

N = 200

# EVOLVE-BLOCK-START

def _distance_matrix(pts):
    """Compute pairwise distances between points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def _compute_radius(pts):
    """Compute the maximum radius for given centers."""
    # Minimum pairwise distance
    dm = _distance_matrix(pts)
    # Exclude diagonal
    np.fill_diagonal(dm, np.inf)
    min_pair = np.min(dm)
    
    # Distance to boundary: for square [-0.5, 0.5]^2
    # For each point, min distance to any of the 4 sides
    dist_to_bounds = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    )
    min_bound = np.min(dist_to_bounds)
    
    return min(min_pair, 2 * min_bound) / 2

def _objective(pts, N=N):
    """Objective to minimize: negative of the radius."""
    # Check validity
    if not np.all(np.isfinite(pts)):
        return 1e6
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        # Points outside square - heavy penalty
        penalty = np.sum(np.maximum(0, np.abs(pts[:, 0]) - 0.5 + 1e-4)) + \
                  np.sum(np.maximum(0, np.abs(pts[:, 1]) - 0.5 + 1e-4))
        return 1e6 + penalty
    
    r = _compute_radius(pts)
    return -r

def _hexagonal_initialization(n_cols):
    """Create a hexagonal lattice initialization."""
    pts = []
    # Hexagonal packing: rows offset by half the horizontal spacing
    h_spacing = 1.0 / n_cols
    v_spacing = h_spacing * np.sqrt(3) / 2
    
    n_rows = int(np.ceil(N / n_cols))
    
    for i in range(n_rows):
        for j in range(n_cols):
            if len(pts) >= N:
                break
            # Offset odd rows by half spacing
            x_offset = 0 if i % 2 == 0 else h_spacing / 2
            x = (j + 0.5) * h_spacing - 0.5 + x_offset
            y = (i + 0.5) * v_spacing - 0.5
            # Clip to stay within bounds
            x = np.clip(x, -0.5, 0.5)
            y = np.clip(y, -0.5, 0.5)
            pts.append([x, y])
    
    pts = np.array(pts[:N])
    return pts

def _grid_initialization(m):
    """Create a grid initialization with m columns."""
    n_rows = int(np.ceil(N / m))
    pts = []
    for i in range(n_rows):
        for j in range(m):
            if len(pts) >= N:
                break
            x = (j + 0.5) / m - 0.5
            y = (i + 0.5) / n_rows - 0.5
            pts.append([x, y])
    pts = np.array(pts[:N])
    return pts

def _random_initialization():
    """Random initialization within the square."""
    return np.random.uniform(-0.5, 0.5, size=(N, 2))

def _jitter(pts, sigma=0.01):
    """Add small random jitter to points."""
    return pts + np.random.normal(0, sigma, size=pts.shape)

def _local_optimize(pts, max_iter=5000, tol=1e-8):
    """Perform local optimization using BFGS."""
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def obj(x):
        pts_x = x.reshape(-1, 2)
        return _objective(pts_x)
    
    x0 = pts.flatten()
    
    try:
        result = minimize(
            obj, x0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6}
        )
        if result.success or result.status == 1:
            best_pts = result.x.reshape(-1, 2)
            # Ensure points are within bounds
            best_pts = np.clip(best_pts, -0.5, 0.5)
            return best_pts, -_objective(best_pts)
        else:
            return pts, _compute_radius(pts)
    except Exception:
        return pts, _compute_radius(pts)

def _simulated_annealing(pts, n_iterations=5000, t0=0.05, t1=0.001):
    """Simulated annealing for global optimization."""
    current_pts = pts.copy()
    current_r = _compute_radius(current_pts)
    best_pts = current_pts.copy()
    best_r = current_r
    
    # Cooling schedule
    alpha = (t1 / t0) ** (1.0 / n_iterations)
    temperature = t0
    
    for it in range(n_iterations):
        # Random move
        idx = np.random.randint(0, N)
        move = np.random.normal(0, temperature, size=2)
        new_pts = current_pts.copy()
        new_pts[idx] = np.clip(new_pts[idx] + move, -0.5, 0.5)
        
        new_r = _compute_radius(new_pts)
        delta_r = new_r - current_r
        
        # Accept if better, or with probability exp(delta_r / temperature)
        if delta_r > 0 or np.random.random() < np.exp(delta_r / temperature):
            current_pts = new_pts
            current_r = new_r
            
            if current_r > best_r:
                best_r = current_r
                best_pts = current_pts.copy()
        
        # Cool down
        temperature *= alpha
    
    return best_pts, best_r

def _hill_climbing(pts, n_iterations=100, step_size=0.005):
    """Simple hill climbing with random moves."""
    current_pts = pts.copy()
    current_r = _compute_radius(current_pts)
    
    for _ in range(n_iterations):
        # Try a random move
        idx = np.random.randint(0, N)
        move = np.random.normal(0, step_size, size=2)
        new_pts = current_pts.copy()
        new_pts[idx] = np.clip(new_pts[idx] + move, -0.5, 0.5)
        
        new_r = _compute_radius(new_pts)
        if new_r > current_r:
            current_pts = new_pts
            current_r = new_r
    
    return current_pts, current_r

def construct():
    """Construct the optimal packing of 200 circles."""
    start_time = time.time()
    time_limit = 110  # Leave some margin
    
    best_pts = None
    best_r = 0.0
    
    # Strategy 1: Try different hexagonal and grid configurations
    for n_cols in [13, 14, 15, 16, 17]:
        if time.time() - start_time > time_limit:
            break
        
        # Hexagonal initialization
        pts = _hexagonal_initialization(n_cols)
        if len(pts) == N:
            r = _compute_radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
        
        # Grid initialization
        pts = _grid_initialization(n_cols)
        if len(pts) == N:
            r = _compute_radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    
    # Strategy 2: Local optimization from best initial point
    if best_pts is not None and time.time() - start_time < time_limit - 30:
        opt_pts, opt_r = _local_optimize(best_pts, max_iter=3000)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts
    
    # Strategy 3: Multiple diverse restarts with simulated annealing
    n_restarts = 8
    for i in range(n_restarts):
        if time.time() - start_time > time_limit - 40:
            break
        
        # Diverse initializations
        if i % 3 == 0:
            pts = _hexagonal_initialization(14 + (i % 3))
        elif i % 3 == 1:
            pts = _grid_initialization(14 + (i % 3))
        else:
            pts = _random_initialization()
        
        # Simulated annealing for global exploration
        if time.time() - start_time < time_limit - 20:
            sa_pts, sa_r = _simulated_annealing(pts, n_iterations=3000, t0=0.03, t1=0.0005)
            if sa_r > best_r:
                best_r = sa_r
                best_pts = sa_pts
    
    # Strategy 4: Local optimization on best solution found
    if best_pts is not None and time.time() - start_time < time_limit - 15:
        opt_pts, opt_r = _local_optimize(best_pts, max_iter=4000)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts
    
    # Strategy 5: Final hill climbing polish
    if best_pts is not None and time.time() - start_time < time_limit - 5:
        hc_pts, hc_r = _hill_climbing(best_pts, n_iterations=300, step_size=0.002)
        if hc_r > best_r:
            best_r = hc_r
            best_pts = hc_pts
    
    # Ensure valid output
    if best_pts is None:
        # Fallback: simple grid
        m = 14
        pts = _grid_initialization(m)
        best_pts = pts
    
    # Final validation and clipping
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts

# EVOLVE-BLOCK-END
