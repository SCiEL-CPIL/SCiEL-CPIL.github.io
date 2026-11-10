# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    if d_min == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _objective(points_flat):
    """Objective function to minimize: -log(R)."""
    n = 16
    points = points_flat.reshape(n, 2)
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    
    # Handle edge cases to avoid log(0) or division by zero
    if d_max <= 1e-10 or d_min <= 1e-10:
        return 1e6
    ratio = d_min / d_max
    if ratio <= 0:
        return 1e6
    return -np.log(ratio)

def _generate_hexagonal_grid(n):
    """Generate a hexagonal grid of approximately n points."""
    points = []
    for j in range(4):
        for i in range(4):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            points.append([x, y])
    points = np.array(points)
    points -= np.mean(points, axis=0)
    return points

def _generate_random_grid(n):
    """Generate a random grid-like structure."""
    np.random.seed(42)
    points = np.random.randn(n, 2)
    points -= np.mean(points, axis=0)
    return points

def _generate_square_grid(n):
    """Generate a square grid."""
    side = int(np.ceil(np.sqrt(n)))
    points = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count < n:
                points.append([i, j])
                count += 1
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates of the points.

    """
    n = 16
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    # Initial candidates
    candidates = []
    
    # 1. Hexagonal lattice
    try:
        hex_pts = _generate_hexagonal_grid(n)
        candidates.append(hex_pts)
    except:
        pass

    # 2. Square grid
    try:
        sq_pts = _generate_square_grid(n)
        candidates.append(sq_pts)
    except:
        pass

    # 3. Random points (multiple seeds)
    rng = np.random.default_rng(42)
    for seed in range(5):
        rand_pts = rng.randn(n, 2)
        rand_pts -= np.mean(rand_pts, axis=0)
        candidates.append(rand_pts)
    
    # 4. Circle points
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(circle_pts)

    best_points = None
    best_score = -1.0

    # Evaluate initial candidates
    for init_pts in candidates:
        score = _compute_score(init_pts)
        if score > best_score:
            best_score = score
            best_points = init_pts

    # Multi-start Nelder-Mead optimization
    # Nelder-Mead is a direct search method that does not use gradients.
    # It is robust for small dimensions and non-smooth objectives.
    
    # We will try to optimize from the best few initial candidates
    # and from random perturbations of the best candidate.
    
    # Sort candidates by score descending
    scored_candidates = []
    for c in candidates:
        s = _compute_score(c)
        scored_candidates.append((s, c))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 3 distinct starting points
    start_points = [c for s, c in scored_candidates[:3]]
    
    # Add some random starts
    for _ in range(3):
        rand_start = rng.randn(n, 2)
        rand_start -= np.mean(rand_start, axis=0)
        start_points.append(rand_start)

    for x0 in start_points:
        if time.time() > deadline - 30:
            break
        
        x0_flat = x0.flatten()
        
        try:
            # Nelder-Mead options
            # maxiter: limit iterations to prevent hanging
            # xatol, fatol: tolerances for stopping
            result = minimize(_objective, x0_flat, method='Nelder-Mead', 
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if result.fun < 1e5: # Valid optimization
                opt_pts = result.x.reshape(n, 2)
                score = _compute_score(opt_pts)
                
                if score > best_score:
                    best_score = score
                    best_points = opt_pts
                    
        except Exception:
            continue

    # If we have time, do a local refinement with a different method or more iterations on the best found
    if best_points is not None and time.time() < deadline - 60:
        x0_flat = best_points.flatten()
        try:
            # Try COBYLA which handles constraints well, though we don't have explicit ones.
            # Or just run Nelder-Mead again with tighter tolerances.
            result = minimize(_objective, x0_flat, method='Nelder-Mead', 
                              options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if result.fun < 1e5:
                opt_pts = result.x.reshape(n, 2)
                score = _compute_score(opt_pts)
                if score > best_score:
                    best_score = score
                    best_points = opt_pts
        except Exception:
            pass

    if best_points is None:
        # Absolute fallback
        return np.random.randn(16, 2)

    return best_points

# EVOLVE-BLOCK-END
