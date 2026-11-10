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
    # For 16 points, a 4x4 block of hex lattice is a good candidate
    # Hex lattice points:
    # (i + j/2, j * sqrt(3)/2)
    points = []
    # Try to fit 16 points in a hex shape
    # Rows: 4, 4, 4, 4? Or 3, 4, 4, 3, 2?
    # A simple 4x4 rectangular block of hex lattice:
    # Row 0: 4 points
    # Row 1: 4 points (shifted)
    # Row 2: 4 points
    # Row 3: 4 points
    # Total 16.
    
    # Let's generate a 4x4 grid
    for j in range(4):
        for i in range(4):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            points.append([x, y])
    points = np.array(points)
    
    # Center it
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
    for seed in [42, 123, 456, 789, 101]:
        np.random.seed(seed)
        rand_pts = np.random.randn(n, 2)
        rand_pts -= np.mean(rand_pts, axis=0)
        candidates.append(rand_pts)

    best_points = None
    best_score = -1.0

    for init_pts in candidates:
        if time.time() > deadline - 60:
            break
            
        # Flatten for optimizer
        x0 = init_pts.flatten()
        
        try:
            # Use BFGS as it is robust for small problems
            result = minimize(_objective, x0, method='BFGS', options={'maxiter': 1000, 'gtol': 1e-10})
            
            if result.success or result.fun < 1e5:
                opt_pts = result.x.reshape(n, 2)
                score = _compute_score(opt_pts)
                
                if score > best_score:
                    best_score = score
                    best_points = opt_pts
                    
        except Exception:
            continue

    # If optimization failed or didn't find anything better, try a simple local search on the best candidate
    if best_points is None:
        # Fallback to the best initial candidate
        for init_pts in candidates:
            score = _compute_score(init_pts)
            if score > best_score:
                best_score = score
                best_points = init_pts
    else:
        # Check if initial candidates were better (unlikely but possible)
        for init_pts in candidates:
            score = _compute_score(init_pts)
            if score > best_score:
                best_score = score
                best_points = init_pts

    if best_points is None:
        # Absolute fallback
        return np.random.randn(16, 2)

    return best_points

# EVOLVE-BLOCK-END
