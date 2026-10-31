import numpy as np
import time
import scipy.spatial as spat
import itertools

N = 13

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(points) < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = spat.ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    pts = points
    n = len(pts)
    
    # Efficient triangle area calculation
    # Pre-computed indices for C(13,3) = 286 triangles
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    v1 = pts[j_flat] - pts[i_flat] 
    v2 = pts[k_flat] - pts[i_flat]
    
    # Determinant in 2D
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _generate_base_grid():
    """
    Generate a fixed base grid of 13 points.
    We use a 4x4 grid of integer coordinates (0,0) to (3,3) and remove one corner.
    Removing a corner breaks symmetry and helps form a convex hull that isn't a perfect square,
    potentially allowing better affine transformations to optimize the min-triangle area.
    Alternatively, we can try different corners or even different base configurations.
    Let's stick to (0,0) to (3,3) minus (0,0) for now.
    """
    grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    # Remove one point, e.g., (0,0)
    grid = grid[1:] # This removes the first row (0,0), (0,1), (0,2), (0,3) -> No, grid[0] is (0,0)
    # Let's be explicit
    grid = np.array([
        [0, 0], [0, 1], [0, 2], [0, 3],
        [1, 0], [1, 1], [1, 2], [1, 3],
        [2, 0], [2, 1], [2, 2], [2, 3],
        [3, 0], [3, 1], [3, 2], [3, 3]
    ], dtype=float)
    # Remove (0,0)
    grid = grid[1:]
    # Center the grid to make affine transformations more stable
    grid -= np.mean(grid, axis=0)
    return grid

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    base_grid = _generate_base_grid()
    
    best_score = -1.0
    best_points = None
    
    # Affine transformation parameters:
    # We are optimizing an affine map T(x) = A @ x + b
    # Since the score is invariant to translation and scaling, we can fix b=0 and constrain A.
    # However, shear is important.
    # Let A = [[a, c], [d, b]]
    # To reduce search space, we can fix one parameter, e.g., b=1 (scale in y), and optimize a, c, d.
    # Or we can optimize all 4 and let the normalization handle scale.
    # Let's optimize a, c, d, b with b=1 fixed to reduce dimensionality to 3.
    # Actually, let's just optimize the 4 parameters of A, and the score function is scale invariant.
    # But to avoid singular matrices, we need det(A) != 0.
    
    # Let's use a grid search or differential evolution on the 4 parameters of A.
    # Given the small search space, a grid search might be sufficient and robust.
    # Range for parameters: -2 to 2 seems reasonable for shear and scale.
    
    # Let's try a coarse grid first, then refine.
    # Parameters: a, c, d, b (where A = [[a, c], [d, b]])
    # Fix b = 1 to reduce dimensionality? No, let's keep it general but normalize.
    # Actually, since score is scale invariant, we can fix one parameter, e.g., a=1.
    # Then we optimize c, d, b.
    # Let's fix a=1 and optimize c, d, b in [-2, 2].
    
    # Grid search for c, d, b
    # Number of points per dimension: 20
    # Total evaluations: 20^3 = 8000. This is very fast.
    
    # Let's use 30 points per dimension for better resolution.
    # 30^3 = 27000 evaluations. Still fast.
    
    # We can also try different base grids (remove different corners) and pick the best.
    
    corners_to_remove = [
        [0, 0], [0, 3], [3, 0], [3, 3] # Corners of the 4x4 grid
    ]
    
    for corner in corners_to_remove:
        # Generate base grid with this corner removed
        grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
        # Find index of corner
        idx = np.where((grid[:, 0] == corner[0]) & (grid[:, 1] == corner[1]))[0][0]
        grid = np.delete(grid, idx, axis=0)
        grid -= np.mean(grid, axis=0)
        
        # Optimize affine transformation A = [[1, c], [d, b]]
        # c, d, b in [-2, 2]
        c_vals = np.linspace(-2, 2, 30)
        d_vals = np.linspace(-2, 2, 30)
        b_vals = np.linspace(-2, 2, 30)
        
        # To save time, we can do a coarse search first
        # Let's do a random sampling of 10000 points in the 3D space
        # This is faster than a full grid and likely to find a good solution.
        
        for _ in range(5000):
            if time.time() > deadline - 5.0:
                break
            
            c = np.random.uniform(-2, 2)
            d = np.random.uniform(-2, 2)
            b = np.random.uniform(-2, 2)
            
            A = np.array([[1.0, c], [d, b]])
            
            # Check if A is singular
            if abs(np.linalg.det(A)) < 1e-6:
                continue
                
            points = A @ grid.T
            points = points.T
            
            score = calculate_score(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
    
    # If we found a good point, we can try to polish it with a local search on the 26 variables
    # But since the strategy is to use the affine structure, we might just return the best affine point.
    # However, a local polish on the 26 variables might improve it further.
    # Let's add a simple local polish step.
    
    if best_points is not None and time.time() < deadline - 10.0:
        # Simple local polish: random perturbations
        current_pts = best_points.copy()
        current_score = best_score
        step_size = 0.1
        
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            
            # Random perturbation
            noise = np.random.normal(0, step_size, (N, 2))
            trial_pts = current_pts + noise
            
            trial_score = calculate_score(trial_pts)
            
            if trial_score > current_score:
                current_pts = trial_pts
                current_score = trial_score
                step_size *= 0.9
            else:
                step_size *= 0.8
                
        if current_score > best_score:
            best_score = current_score
            best_points = current_pts
    
    if best_points is None:
        # Fallback to random points
        return np.random.rand(N, 2)
        
    return best_points
