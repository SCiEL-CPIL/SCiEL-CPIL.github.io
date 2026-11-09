# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Strategy:
    1. Generate diverse structural initializations (regular polygons, grids, random perturbations).
    2. Use a smooth soft-min surrogate objective for L-BFGS-B optimization.
    3. Add a high-precision polish phase with tighter tolerances.
    
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

    def calculate_metrics(points):
        """
        Calculate A_min / A_hull and validity.
        Returns (score, A_min, A_hull, is_valid)
        """
        if not np.all(np.isfinite(points)):
            return 0.0, 0.0, 0.0, False
        
        # Calculate convex hull area
        try:
            hull = ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-10:
                return 0.0, 0.0, A_hull, False
        except Exception:
            return 0.0, 0.0, 0.0, False

        # Calculate min triangle area
        p1 = points[i_idx]
        p2 = points[j_idx]
        p3 = points[k_idx]
        
        # Cross product magnitude in 2D
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        
        A_min = np.min(areas)
        
        if A_min <= 1e-10:
            return 0.0, A_min, A_hull, True
            
        score = A_min / A_hull
        return score, A_min, A_hull, True

    def soft_min_objective(points_flat, temperature=0.01):
        """
        Smooth surrogate: minimize -soft_min(areas) / A_hull
        soft_min(areas) = -T * log(sum(exp(-areas/T)))
        As T -> 0, soft_min -> min
        """
        points = points_flat.reshape((n, 2))
        
        if not np.all(np.isfinite(points)):
            return 1e6
        
        # Calculate convex hull area
        try:
            hull = ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-10:
                return 1e6
        except Exception:
            return 1e6

        # Calculate all triangle areas
        p1 = points[i_idx]
        p2 = points[j_idx]
        p3 = points[k_idx]
        
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        
        # Soft-min approximation
        # To avoid overflow, shift by max
        max_area = np.max(areas)
        shifted = -areas / temperature
        shifted -= np.max(shifted)  # for numerical stability
        log_sum_exp = np.log(np.sum(np.exp(shifted)))
        soft_min_val = -temperature * (log_sum_exp + np.max(-areas / temperature))
        
        # We want to maximize soft_min_val / A_hull
        # So minimize -soft_min_val / A_hull
        if soft_min_val <= 1e-10:
            return 1e6
            
        return -soft_min_val / A_hull

    def objective_direct(points_flat):
        """
        Direct objective: minimize -A_min / A_hull
        """
        points = points_flat.reshape((n, 2))
        score, A_min, A_hull, is_valid = calculate_metrics(points)
        
        if not is_valid:
            return 1e6
        return -score

    def optimize_from_start(start_points, max_iter=200, use_soft_min=True, temperature=0.01):
        """
        Run L-BFGS-B from a specific starting point.
        """
        if time.time() - start_time > time_limit:
            return None, 0.0
            
        start_flat = start_points.flatten()
        
        # Bounds: Keep points within a reasonable box to avoid drift
        bounds = [(-10.0, 10.0)] * (n * 2)
        
        if use_soft_min:
            obj_func = lambda x: soft_min_objective(x, temperature=temperature)
        else:
            obj_func = objective_direct
        
        try:
            res = minimize(
                obj_func, 
                start_flat, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            best_points = res.x.reshape((n, 2))
            score, _, _, is_valid = calculate_metrics(best_points)
            if is_valid and score > 0:
                return best_points, score
        except Exception:
            pass
            
        return None, 0.0

    def polish(points, max_iter=500):
        """
        High-precision polish with very tight tolerances.
        """
        if time.time() - start_time > time_limit - 30:
            return points, 0.0
        
        start_flat = points.flatten()
        bounds = [(-10.0, 10.0)] * (n * 2)
        
        # Use direct objective for polish, with very tight tolerances
        try:
            res = minimize(
                objective_direct, 
                start_flat, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            polished = res.x.reshape((n, 2))
            score, _, _, is_valid = calculate_metrics(polished)
            if is_valid and score > 0:
                return polished, score
        except Exception:
            pass
            
        return points, 0.0

    # --- Generate Initial Structures ---
    
    structures = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 1.0
    regular_13 = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    structures.append(regular_13)
    
    # 2. Regular 7-gon + 6 points inside
    angles7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer7 = np.column_stack([1.0 * np.cos(angles7), 1.0 * np.sin(angles7)])
    angles6 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    inner6 = np.column_stack([0.5 * np.cos(angles6), 0.5 * np.sin(angles6)])
    structures.append(np.vstack([outer7, inner6]))
    
    # 3. Regular 9-gon + 4 points in a square
    angles9 = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    outer9 = np.column_stack([1.0 * np.cos(angles9), 1.0 * np.sin(angles9)])
    sq_size = 0.4
    square_pts = np.array([
        [-sq_size, -sq_size],
        [sq_size, -sq_size],
        [sq_size, sq_size],
        [-sq_size, sq_size]
    ])
    structures.append(np.vstack([outer9, square_pts]))
    
    # 4. Regular 12-gon + 1 center
    angles12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    outer12 = np.column_stack([1.0 * np.cos(angles12), 1.0 * np.sin(angles12)])
    center = np.array([[0.0, 0.0]])
    structures.append(np.vstack([outer12, center]))
    
    # 5. Grid-like structure
    grid_pts = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2],
        [1, 3]
    ])
    grid_pts = (grid_pts - 1.5) * 1.0
    structures.append(grid_pts)
    
    # 6. Hexagonal packing pattern
    hex_pts = []
    # Center
    hex_pts.append([0, 0])
    # First ring (6 points)
    for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
        hex_pts.append([0.8*np.cos(angle), 0.8*np.sin(angle)])
    # Second ring (6 more points, offset)
    for angle in np.linspace(np.pi/6, 2*np.pi + np.pi/6, 6, endpoint=False):
        hex_pts.append([1.4*np.cos(angle), 1.4*np.sin(angle)])
    hex_pts = np.array(hex_pts)
    structures.append(hex_pts)
    
    # 7. Random perturbations of the regular 13-gon
    rng = np.random.default_rng(seed=42)
    for _ in range(15):
        perturbed = regular_13 + rng.normal(0, 0.15, (n, 2))
        structures.append(perturbed)
        
    # 8. Random points in a circle
    for _ in range(15):
        r = rng.uniform(0.4, 1.0, n)
        theta = rng.uniform(0, 2 * np.pi, n)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        structures.append(pts)
    
    # 9. Two concentric 7-gons with offset
    for offset in [0.3, 0.5, 0.7]:
        angles7a = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        angles7b = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/7
        outer = np.column_stack([1.0 * np.cos(angles7a), 1.0 * np.sin(angles7a)])
        inner = np.column_stack([offset * np.cos(angles7b), offset * np.sin(angles7b)])
        structures.append(np.vstack([outer, inner]))
    
    # 10. Square with points inside
    for sq_size in [0.3, 0.5, 0.7]:
        outer_sq = np.array([
            [-1, -1], [1, -1], [1, 1], [-1, 1],
            [0, -1], [1, 0], [0, 1], [-1, 0]
        ])
        inner_pts = rng.uniform(-sq_size, sq_size, (5, 2))
        structures.append(np.vstack([outer_sq, inner_pts]))

    # --- Optimization Loop ---
    
    best_score = 0.0
    best_points = None
    
    # First pass: use soft-min objective for smoother landscape
    for i, start_pts in enumerate(structures):
        if time.time() - start_time > time_limit - 120:  # Reserve 120s for polish
            break
            
        # Refine this structure with soft-min
        refined_pts, score = optimize_from_start(start_pts, max_iter=150, use_soft_min=True, temperature=0.05)
        
        if refined_pts is not None and score > best_score:
            best_score = score
            best_points = refined_pts
            
    # Second pass: refine top candidates with direct objective
    if best_points is not None and time.time() - start_time < time_limit - 60:
        # Refine the best point with direct objective
        refined_pts, score = optimize_from_start(best_points, max_iter=200, use_soft_min=False)
        if refined_pts is not None and score > best_score:
            best_score = score
            best_points = refined_pts
    
    # --- Final Polish Phase ---
    # High-precision polish with very tight tolerances
    if best_points is not None and time.time() - start_time < time_limit - 20:
        polished_pts, polished_score = polish(best_points, max_iter=300)
        if polished_score > best_score:
            best_points = polished_pts
            best_score = polished_score
            
    # --- Normalization ---
    # The score is invariant to scaling and translation, but we should ensure 
    # the points are well-scaled for the evaluator.
    if best_points is not None:
        centroid = np.mean(best_points, axis=0)
        best_points = best_points - centroid
        max_dist = np.max(np.linalg.norm(best_points, axis=1))
        if max_dist > 0:
            best_points = best_points / max_dist
            
        # Final verification
        final_score, _, _, is_valid = calculate_metrics(best_points)
        if is_valid and final_score > 0:
            return best_points
    
    # Fallback: Return a regular 13-gon if everything failed
    return regular_13

# EVOLVE-BLOCK-END
