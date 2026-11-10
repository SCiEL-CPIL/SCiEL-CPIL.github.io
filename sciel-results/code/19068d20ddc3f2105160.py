import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Returns 16 points in 2D that maximize the min/max distance ratio.
    Uses multiple initializations (grid, hexagonal, random) and Powell optimization
    to refine the point positions.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    n = 16
    
    def objective(params):
        """
        Objective function to minimize: -(d_min / d_max)
        """
        pts = params.reshape((n, 2))
        dists = pdist(pts, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-10:
            return 0.0
            
        ratio = d_min / d_max
        return -ratio

    def get_score(pts):
        dists = pdist(pts, 'euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    best_points = None
    best_score = -np.inf
    
    # Define initial configurations
    
    # 1. 4x4 Grid
    x = np.arange(4)
    y = np.arange(4)
    xx, yy = np.meshgrid(x, y)
    grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
    
    # 2. Hexagonal Lattice Subset
    u = np.array([1.0, 0.0])
    v = np.array([0.5, np.sqrt(3)/2])
    points_list = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * u + j * v
            dist_sq = np.dot(p, p)
            points_list.append((dist_sq, p))
    points_list.sort(key=lambda x: x[0])
    hex_pts = np.array([p[1] for p in points_list[:16]])
    
    # 3. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 1.0
    circle_pts = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    
    # 4. Random points (seeded for reproducibility)
    np.random.seed(42)
    rand_pts = np.random.randn(n, 2) * 2
    
    initial_guesses = [grid_pts, hex_pts, circle_pts, rand_pts]
    
    # Optimization Loop
    for i, init_pts in enumerate(initial_guesses):
        if time.time() > deadline:
            break
            
        # Normalize initial points to roughly unit scale to help optimizer
        # The ratio is scale invariant, but numerical stability is better with O(1) values
        scale = np.max(np.linalg.norm(init_pts, axis=1))
        if scale > 0:
            init_pts_scaled = init_pts / scale
        else:
            init_pts_scaled = init_pts
            
        try:
            # Use Powell's method for robustness against non-smooth objective
            res = minimize(
                objective, 
                init_pts_scaled.flatten(), 
                method='Powell', 
                options={'maxiter': 500, 'ftol': 1e-10, 'xtol': 1e-10}
            )
            
            if time.time() > deadline:
                # Check if current best is valid before returning
                if best_points is not None:
                    return best_points
                return res.x.reshape((n, 2))
                
            candidate_points = res.x.reshape((n, 2))
            
            # Calculate score for this candidate
            candidate_score = get_score(candidate_points)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points.copy()
                
        except Exception:
            continue

    # If no valid points were found (should not happen), return grid
    if best_points is None:
        return grid_pts

    return best_points
