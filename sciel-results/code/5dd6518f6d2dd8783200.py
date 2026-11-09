import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 580  # Leave 50s margin from 630s to be safe

    # Pre-compute indices for all 286 triangles (i < j < k)
    tri_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    i_idx = tri_indices[:, 0]
    j_idx = tri_indices[:, 1]
    k_idx = tri_indices[:, 2]

    def calculate_score(points):
        """
        Calculate A_min / A_hull. Returns negative for invalid or 0.
        points: shape (13, 2)
        """
        if not np.all(np.isfinite(points)):
            return 0.0

        # Calculate convex hull area
        try:
            hull = ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-12:
                return 0.0
        except Exception:
            return 0.0

        # Calculate min triangle area
        p1 = points[i_idx]
        p2 = points[j_idx]
        p3 = points[k_idx]
        
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        
        A_min = np.min(areas)
        
        if A_min <= 0:
            return 0.0
            
        return A_min / A_hull

    def neg_objective(points_flat):
        """
        Negated objective for minimization.
        """
        points = points_flat.reshape((n, 2))
        score = calculate_score(points)
        if score == 0.0:
            return 1e6
        return -score

    best_points = None
    best_score = -np.inf

    # Strategy:
    # 1. Generate candidate points on integer grids of various sizes (e.g., 4x4, 5x5, 6x6).
    # 2. For each grid, we cannot check all C(25, 13) combos. Instead, we use a heuristic:
    #    - Try to select points that are well-spread.
    #    - Or, simply sample random subsets of the grid points.
    #    - Or, use a greedy algorithm to add points that maximize the min distance to existing points.
    #    
    # Given the time limit, a random sampling of grid subsets is likely too slow if we do many.
    # Let's try a few specific structured patterns first, then random grid subsets.
    
    candidates = []
    
    # 1. Regular 13-gon (baseline)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 0.5
    cx, cy = 0.5, 0.5
    init_x = cx + radius * np.cos(angles)
    init_y = cy + radius * np.sin(angles)
    init_points = np.column_stack((init_x, init_y))
    candidates.append(init_points)
    
    # 2. Grid-based candidates
    # We will try grids of size GxG where G is 4, 5, 6.
    # For each grid, we will generate a few random subsets of 13 points.
    # To make it faster, we pre-compute all points on the grid.
    
    rng = np.random.default_rng(42)
    
    for G in [4, 5, 6, 7]:
        # Create grid points
        grid_x, grid_y = np.meshgrid(np.arange(G), np.arange(G))
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        num_grid_pts = len(grid_points)
        
        if num_grid_pts < n:
            continue
            
        # Try a few random subsets
        # Number of subsets to try depends on time. 
        # Let's try 100 random subsets per grid size.
        num_samples = 100
        for _ in range(num_samples):
            if time.time() - start_time > time_limit - 60: # Reserve 60s for optimization
                break
            # Randomly pick 13 distinct points from the grid
            indices = rng.choice(num_grid_pts, size=n, replace=False)
            subset = grid_points[indices]
            
            # Normalize to [0,1] for consistent scaling? 
            # The score is scale invariant, but L-BFGS-B needs bounds.
            # Let's keep them in [0, G-1] for now, and let the optimizer handle it.
            # Actually, for the initial score calculation, scale doesn't matter.
            candidates.append(subset)
            
    # 3. Add some perturbed regular polygons
    for _ in range(10):
        perturbed = init_points + rng.normal(0, 0.1, (n, 2))
        # Clip to [0, 1] roughly? No, let's just use it.
        candidates.append(perturbed)

    # Evaluate all candidates
    scored_candidates = []
    for pts in candidates:
        score = calculate_score(pts)
        scored_candidates.append((score, pts))
        
    # Sort by score descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take the top K candidates for local optimization
    top_k = 10
    top_candidates = [pts for score, pts in scored_candidates[:top_k] if score > 0]
    
    # If no valid candidates, use the regular polygon
    if not top_candidates:
        top_candidates = [init_points]
        
    # Local optimization on the top candidates
    bounds = [(0.0, 10.0)] * (n * 2) # Wider bounds to allow grid points to move
    
    for start_pts in top_candidates:
        if time.time() - start_time > time_limit:
            break
            
        # Normalize start points to [0, 1] to help the optimizer?
        # The score is scale invariant, so the optimizer might drift in scale.
        # To prevent this, we can fix the centroid and scale in the objective, 
        # but that requires modifying the objective. 
        # Instead, let's just run the optimizer on the raw coordinates.
        # If the scale drifts, the score remains the same, so it's fine.
        
        try:
            result = minimize(
                neg_objective, 
                start_pts.flatten(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            current_points = result.x.reshape((n, 2))
            current_score = calculate_score(current_points)
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points
                
        except Exception:
            continue
            
    # If best_points is still None or invalid, fallback
    if best_points is None or not np.all(np.isfinite(best_points)):
         init_score = calculate_score(init_points)
         if init_score > 0:
             best_points = init_points
         else:
             best_points = rng.random((n, 2))
             
    return best_points
