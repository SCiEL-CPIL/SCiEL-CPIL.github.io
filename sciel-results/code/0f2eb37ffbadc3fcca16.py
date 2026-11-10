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

def _min_dist(points_flat):
    """Compute minimum pairwise distance."""
    n = 16
    points = points_flat.reshape(n, 2)
    d = pdist(points)
    return np.min(d)

def _max_dist(points_flat):
    """Compute maximum pairwise distance."""
    n = 16
    points = points_flat.reshape(n, 2)
    d = pdist(points)
    return np.max(d)

def _objective_minimize_neg_min_dist(x):
    """Objective: minimize negative of minimum distance."""
    return -_min_dist(x)

def _constraint_max_dist(x, d_max_bound):
    """Constraint: max distance <= d_max_bound."""
    return d_max_bound - _max_dist(x)

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

def _generate_circle(n):
    """Generate points on a circle."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points

def _generate_random(n, seed=None):
    """Generate random points."""
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n, 2))
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

def _optimize_from_init(x0, deadline):
    """Optimize from initial point using SLSQP with multiple d_max bounds."""
    n = 16
    best_points = x0.copy()
    best_score = _compute_score(x0)
    
    # Get current distances to set reasonable bounds
    current_d_max = _max_dist(x0)
    current_d_min = _min_dist(x0)
    
    # Try several d_max bounds to explore different scales
    for scale in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
        if time.time() > deadline - 10:
            break
        
        d_max_bound = current_d_max * scale
        
        try:
            result = minimize(
                _objective_minimize_neg_min_dist,
                x0.copy(),
                method='SLSQP',
                constraints=[{'type': 'ineq', 'fun': lambda x, bound=d_max_bound: _constraint_max_dist(x, bound)}],
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < -1e-10:
                opt_pts = result.x.reshape(n, 2)
                if np.all(np.isfinite(opt_pts)):
                    score = _compute_score(opt_pts)
                    if score > best_score:
                        best_score = score
                        best_points = opt_pts
        except Exception:
            continue
    
    return best_points, best_score

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the points.

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

    # 3. Circle points
    try:
        circ_pts = _generate_circle(n)
        candidates.append(circ_pts)
    except:
        pass

    # 4. Random points (multiple seeds)
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        try:
            rand_pts = _generate_random(n, seed)
            candidates.append(rand_pts)
        except:
            pass

    best_points = None
    best_score = -1.0

    # Evaluate initial candidates
    for init_pts in candidates:
        score = _compute_score(init_pts)
        if score > best_score:
            best_score = score
            best_points = init_pts

    # Sort candidates by score descending
    scored_candidates = []
    for c in candidates:
        s = _compute_score(c)
        scored_candidates.append((s, c))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 5 distinct starting points for optimization
    start_points = [c for s, c in scored_candidates[:5]]
    
    # Add some random starts
    rng = np.random.default_rng(999)
    for _ in range(3):
        rand_start = rng.standard_normal((n, 2))
        rand_start -= np.mean(rand_start, axis=0)
        start_points.append(rand_start)

    # Optimize from each starting point
    for x0 in start_points:
        if time.time() > deadline - 30:
            break
        
        try:
            opt_pts, opt_score = _optimize_from_init(x0, deadline)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts
        except Exception:
            continue

    # If we have time, do additional refinement on the best found
    if best_points is not None and time.time() < deadline - 60:
        try:
            opt_pts, opt_score = _optimize_from_init(best_points, deadline)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts
        except Exception:
            pass

    if best_points is None:
        # Absolute fallback
        return _generate_random(n, 42)

    # Ensure finite values and correct shape
    if not np.all(np.isfinite(best_points)):
        best_points = np.where(np.isfinite(best_points), best_points, 0.0)
    
    return best_points

# EVOLVE-BLOCK-END
