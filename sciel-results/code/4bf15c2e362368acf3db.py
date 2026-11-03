"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _compute_all_triangle_areas(points):
    """
    Compute areas of all C(n,3) triangles and the convex hull area.
    Returns (areas array, hull_area, min_area_idx) where min_area_idx is the index
    in the flattened triangle list of the minimum area triangle.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = points.shape[0]
    if n < 3:
        return np.array([0.0]), 1.0, 0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return np.array([0.0]), 1.0, 0
    
    if hull_area <= 1e-10:
        return np.array([0.0]), 1.0, 0
    
    # Generate all index combinations i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_vals = i_idx[mask]
    j_vals = j_idx[mask]
    k_vals = k_idx[mask]
    
    # Compute areas vectorized
    p1 = points[i_vals]
    p2 = points[j_vals]
    p3 = points[k_vals]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_idx = np.argmin(areas)
    
    return areas, hull_area, min_idx


def _min_triangle_area_over_hull(points):
    """
    Compute min triangle area / convex hull area for given points.
    Returns the ratio (higher is better).
    """
    areas, hull_area, min_idx = _compute_all_triangle_areas(points)
    if len(areas) == 0 or hull_area <= 1e-10:
        return 0.0
    min_area = np.min(areas)
    if min_area < 1e-10:
        return 0.0
    return min_area / hull_area


def _push_pull_optimize(initial_points, max_time=60.0, rng=None):
    """
    Directed local search: repeatedly find the worst (smallest area) triangle
    and push its vertices outward to increase its area, while monitoring
    that other triangles don't become worse.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    start_time = time.time()
    
    current_points = initial_points.copy()
    areas, hull_area, min_idx = _compute_all_triangle_areas(current_points)
    if len(areas) == 0:
        return current_points, 0.0
    
    current_score = np.min(areas) / hull_area
    best_points = current_points.copy()
    best_score = current_score
    
    # Get the indices of the worst triangle
    n = current_points.shape[0]
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_vals = i_idx[mask]
    j_vals = j_idx[mask]
    k_vals = k_idx[mask]
    
    # Number of iterations
    max_iter = 20000
    
    for it in range(max_iter):
        if time.time() - start_time > max_time:
            break
        
        # Recompute areas and find worst triangle
        areas, hull_area, min_idx = _compute_all_triangle_areas(current_points)
        if len(areas) == 0:
            break
        
        current_score = np.min(areas) / hull_area
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
        
        # Get the worst triangle's vertex indices
        tri_i = i_vals[min_idx]
        tri_j = j_vals[min_idx]
        tri_k = k_vals[min_idx]
        
        p_i = current_points[tri_i]
        p_j = current_points[tri_j]
        p_k = current_points[tri_k]
        
        # Compute centroid of the worst triangle
        tri_centroid = (p_i + p_j + p_k) / 3.0
        
        # Compute the overall centroid
        overall_centroid = np.mean(current_points, axis=0)
        
        # Direction to push: away from overall centroid, but also consider
        # the triangle's own geometry
        push_dir = tri_centroid - overall_centroid
        push_norm = np.linalg.norm(push_dir)
        if push_norm < 1e-10:
            push_dir = rng.randn(2)
            push_norm = np.linalg.norm(push_dir)
        push_dir = push_dir / push_norm
        
        # Try multiple step sizes for the push
        best_candidate = None
        best_candidate_score = current_score
        
        base_step = 0.02
        for step_mult in [0.5, 1.0, 1.5, 2.0, 3.0]:
            step = base_step * step_mult
            
            new_points = current_points.copy()
            
            # Push each vertex of the worst triangle outward
            for vertex_idx, vertex in [(tri_i, p_i), (tri_j, p_j), (tri_k, p_k)]:
                # Direction from overall centroid to this vertex
                v_dir = vertex - overall_centroid
                v_norm = np.linalg.norm(v_dir)
                if v_norm < 1e-10:
                    v_dir = rng.randn(2)
                    v_norm = np.linalg.norm(v_dir)
                v_dir = v_dir / v_norm
                
                # Also add a component perpendicular to the triangle edge
                # to maintain spread
                new_points[vertex_idx] = vertex + step * (0.7 * v_dir + 0.3 * push_dir)
            
            # Compute new score
            new_areas, new_hull_area, _ = _compute_all_triangle_areas(new_points)
            if len(new_areas) == 0 or new_hull_area <= 1e-10:
                continue
            
            new_score = np.min(new_areas) / new_hull_area
            
            if new_score > best_candidate_score:
                best_candidate_score = new_score
                best_candidate = new_points.copy()
        
        if best_candidate is not None:
            current_points = best_candidate
        else:
            # No improvement found with push; try a small random perturbation
            # of the worst triangle vertices
            improved = False
            for _ in range(5):
                new_points = current_points.copy()
                for vertex_idx in [tri_i, tri_j, tri_k]:
                    noise = 0.01 * rng.randn(2)
                    new_points[vertex_idx] += noise
                
                new_areas, new_hull_area, _ = _compute_all_triangle_areas(new_points)
                if len(new_areas) == 0 or new_hull_area <= 1e-10:
                    continue
                
                new_score = np.min(new_areas) / new_hull_area
                if new_score > current_score:
                    current_points = new_points
                    improved = True
                    break
            
            if not improved:
                # Stuck: try a larger random restart of just the worst triangle
                new_points = current_points.copy()
                for vertex_idx in [tri_i, tri_j, tri_k]:
                    # Move to a random position in a small disk around current pos
                    angle = rng.uniform(0, 2 * np.pi)
                    radius = rng.uniform(0.05, 0.15)
                    new_points[vertex_idx] = current_points[vertex_idx] + radius * np.array([np.cos(angle), np.sin(angle)])
                
                new_areas, new_hull_area, _ = _compute_all_triangle_areas(new_points)
                if len(new_areas) == 0 or new_hull_area <= 1e-10:
                    continue
                
                new_score = np.min(new_areas) / new_hull_area
                # Accept even if slightly worse to escape local optima (annealing-like)
                if new_score > current_score * 0.95:
                    current_points = new_points
        
        # Check convergence
        if it > 100 and abs(current_score - best_score) < 1e-8:
            # Try a few more iterations with smaller steps
            base_step = 0.005
    
    return best_points, best_score


def construct():
    """
    Uses directed push-pull local search with multiple initial configurations.
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
    angles2 = 2 * np.pi * (np.arange(N) + rng.rand(N) * 0.03) / N
    points2 = np.column_stack([np.cos(angles2), np.sin(angles2)])
    try:
        hull = ConvexHull(points2)
        area = hull.volume
        if area > 0:
            points2 /= np.sqrt(area)
    except:
        pass
    initial_configs.append(points2.copy())
    
    # 3. Points on two concentric circles (inner and outer)
    n_outer = 8
    n_inner = N - n_outer
    outer_angles = 2 * np.pi * np.arange(n_outer) / n_outer
    inner_angles = 2 * np.pi * (np.arange(n_inner) + 0.5 / n_inner) / n_inner
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)]) * 1.0
    inner_pts = np.column_stack([np.cos(inner_angles), np.sin(inner_angles)]) * 0.5
    points3 = np.vstack([outer_pts, inner_pts])
    try:
        hull = ConvexHull(points3)
        area = hull.volume
        if area > 0:
            points3 /= np.sqrt(area)
    except:
        pass
    initial_configs.append(points3.copy())
    
    # 4. Random points in unit disk
    angles_rand = rng.uniform(0, 2 * np.pi, N)
    radii_rand = rng.uniform(0.3, 1.0, N)
    points4 = np.column_stack([radii_rand * np.cos(angles_rand), radii_rand * np.sin(angles_rand)])
    try:
        hull = ConvexHull(points4)
        area = hull.volume
        if area > 0:
            points4 /= np.sqrt(area)
    except:
        pass
    initial_configs.append(points4.copy())
    
    # 5. Fibonacci spiral-like distribution
    golden_angle = np.pi * (3 - np.sqrt(5))
    points5 = np.zeros((N, 2))
    for i in range(N):
        r = np.sqrt(i / N)
        theta = i * golden_angle
        points5[i] = [r * np.cos(theta), r * np.sin(theta)]
    points5 /= np.linalg.norm(points5, axis=1, keepdims=True) * 0 + 1  # Already in unit disk
    try:
        hull = ConvexHull(points5)
        area = hull.volume
        if area > 0:
            points5 /= np.sqrt(area)
    except:
        pass
    initial_configs.append(points5.copy())
    
    # Run push-pull from each initial config with time budget
    best_overall = initial_configs[0].copy()
    best_score_overall = _min_triangle_area_over_hull(best_overall)
    
    # Allocate time: ~12s per config, 5 configs = 60s, leaving 60s buffer
    time_per_config = 12.0
    
    for i, init_pts in enumerate(initial_configs):
        seed = 42 + i * 1000
        rng_i = np.random.RandomState(seed)
        
        result_pts, result_score = _push_pull_optimize(init_pts, max_time=time_per_config, rng=rng_i)
        
        if result_score > best_score_overall:
            best_score_overall = result_score
            best_overall = result_pts.copy()
    
    # Final polish: run one more optimization from the best found with longer time
    rng_final = np.random.RandomState(999)
    result_pts, result_score = _push_pull_optimize(best_overall, max_time=50.0, rng=rng_final)
    
    if result_score > best_score_overall:
        best_score_overall = result_score
        best_overall = result_pts.copy()
    
    return best_overall
# EVOLVE-BLOCK-END
