"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _min_triangle_area_over_hull(points):
    """
    Compute min triangle area / convex hull area for given points.
    Returns the ratio (higher is better).
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
    
    if hull_area <= 1e-10:
        return 0.0
    
    # Vectorized triangle area calculation using all triples
    # For N=14, there are C(14,3) = 364 triangles
    min_area = float('inf')
    
    # Use a more efficient vectorized approach
    # Generate all index combinations
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    # Filter for i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_vals = i_idx[mask]
    j_vals = j_idx[mask]
    k_vals = k_idx[mask]
    
    # Compute areas vectorized
    p1 = points[i_vals]
    p2 = points[j_vals]
    p3 = points[k_vals]
    
    # Cross product magnitude / 2
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area < 1e-10:
        return 0.0
    
    return min_area / hull_area


def _sa_optimize(rng, initial_points, max_time=90.0):
    """
    Run Simulated Annealing with adaptive step size and restarts.
    """
    import time
    
    start_time = time.time()
    
    current_points = initial_points.copy()
    current_score = _min_triangle_area_over_hull(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    # SA Parameters
    T_start = 0.03
    T_end = 1e-6
    total_iterations = 80000
    cooling_rate = (T_end / T_start) ** (1.0 / total_iterations)
    
    current_T = T_start
    step_size = 0.08
    
    # Adaptive step size tracking
    no_improve_count = 0
    max_no_improve = 5000
    
    for it in range(total_iterations):
        # Check time limit
        if time.time() - start_time > max_time:
            break
        
        # Adaptive step size: shrink if too many non-improvements
        if no_improve_count > max_no_improve:
            step_size *= 0.7
            no_improve_count = 0
            # Occasionally reheat
            if rng.rand() < 0.1:
                current_T = max(current_T * 2, T_start * 0.1)
        
        # Generate perturbation
        # Mix of single-point and multi-point moves
        move_type = rng.random()
        if move_type < 0.6:
            num_perturb = 1
        elif move_type < 0.85:
            num_perturb = 2
        else:
            num_perturb = rng.randint(3, 5)
        
        indices = rng.choice(N, size=num_perturb, replace=False)
        
        new_points = current_points.copy()
        # Use correlated noise for better exploration
        base_noise = step_size * rng.randn(2)
        for idx in indices:
            # Each point gets noise scaled by its distance from centroid (larger for outer points)
            centroid = np.mean(current_points, axis=0)
            dist = np.linalg.norm(current_points[idx] - centroid)
            scale = 1.0 + 0.3 * dist
            noise = step_size * scale * rng.randn(2)
            new_points[idx] += noise
        
        new_score = _min_triangle_area_over_hull(new_points)
        
        # Metropolis acceptance
        delta = new_score - current_score
        if delta > 0:
            current_points = new_points
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            if delta > -50 * current_T:
                accept_prob = np.exp(delta / current_T)
            else:
                accept_prob = 0.0
            
            if rng.rand() < accept_prob:
                current_points = new_points
                current_score = new_score
                no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Cool down
        current_T *= cooling_rate
    
    return best_points, best_score


def construct():
    """
    Uses multi-restart Simulated Annealing with adaptive step sizes.
    """
    rng = np.random.RandomState(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    try:
        hull = ConvexHull(points)
        area = hull.volume
        if area > 0:
            points /= np.sqrt(area)
    except:
        pass
    initial_configs.append(points.copy())
    
    # 2. Regular polygon with slight random perturbation
    angles2 = 2 * np.pi * (np.arange(N) + rng.rand(N) * 0.05) / N
    points2 = np.column_stack([np.cos(angles2), np.sin(angles2)])
    try:
        hull = ConvexHull(points2)
        area = hull.volume
        if area > 0:
            points2 /= np.sqrt(area)
    except:
        pass
    initial_configs.append(points2.copy())
    
    # 3. Random points in unit disk, then projected to convex position
    # Use a mix of boundary and interior points
    n_boundary = 8
    n_interior = N - n_boundary
    boundary_angles = np.sort(rng.uniform(0, 2 * np.pi, n_boundary))
    boundary_pts = np.column_stack([np.cos(boundary_angles), np.sin(boundary_angles)])
    interior_pts = rng.uniform(-0.5, 0.5, size=(n_interior, 2))
    points3 = np.vstack([boundary_pts, interior_pts])
    try:
        hull = ConvexHull(points3)
        area = hull.volume
        if area > 0:
            points3 /= np.sqrt(area)
    except:
        pass
    initial_configs.append(points3.copy())
    
    # 4. Another random configuration
    points4 = rng.uniform(-1, 1, size=(N, 2))
    try:
        hull = ConvexHull(points4)
        area = hull.volume
        if area > 0:
            points4 /= np.sqrt(area)
    except:
        pass
    initial_configs.append(points4.copy())
    
    # Run SA from each initial config with time budget
    best_overall = initial_configs[0].copy()
    best_score_overall = _min_triangle_area_over_hull(best_overall)
    
    # Allocate time: ~20s per config, 4 configs = 80s, leaving 40s buffer
    time_per_config = 18.0
    
    for i, init_pts in enumerate(initial_configs):
        seed = 42 + i * 1000
        rng_i = np.random.RandomState(seed)
        
        result_pts, result_score = _sa_optimize(rng_i, init_pts, max_time=time_per_config)
        
        if result_score > best_score_overall:
            best_score_overall = result_score
            best_overall = result_pts.copy()
    
    # Final polish: run one more SA from the best found
    rng_final = np.random.RandomState(999)
    result_pts, result_score = _sa_optimize(rng_final, best_overall, max_time=30.0)
    
    if result_score > best_score_overall:
        best_score_overall = result_score
        best_overall = result_pts.copy()
    
    return best_overall
# EVOLVE-BLOCK-END
