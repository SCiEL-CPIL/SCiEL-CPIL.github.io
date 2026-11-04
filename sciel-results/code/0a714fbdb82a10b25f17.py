import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 14
# Pre-compute indices for all triangles i < j < k
TRI_INDICES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area using pre-computed indices."""
    if len(points) < 3:
        return 0.0
    i, j, k = TRI_INDICES.T
    pi = points[i]
    pj = points[j]
    pk = points[k]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _build_ring_config(radii, angles_offsets, n_per_ring):
    """
    Build points from concentric rings with rotational symmetry.
    radii: array of ring radii
    angles_offsets: array of angular offsets for each ring
    n_per_ring: array of number of points per ring
    Returns: (N, 2) array of points
    """
    points_list = []
    total = 0
    for idx, (r, offset, n) in enumerate(zip(radii, angles_offsets, n_per_ring)):
        if n == 0:
            continue
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + offset
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        points_list.append(np.column_stack((x, y)))
        total += n
    if total != N:
        raise ValueError(f"Expected {N} points, got {total}")
    return np.vstack(points_list)

def _param_to_points(params, ring_sizes):
    """
    Convert parameter vector to points.
    params: [r1, r2, ..., r_k, theta1, theta2, ..., theta_k]
    ring_sizes: list of n_i for each ring
    """
    k = len(ring_sizes)
    radii = params[:k]
    offsets = params[k:]
    return _build_ring_config(radii, offsets, ring_sizes)

def _score_ring_config(params, ring_sizes):
    """Score for a ring configuration."""
    try:
        pts = _param_to_points(params, ring_sizes)
        return _score(pts)
    except Exception:
        return 0.0

def _neg_score_ring_config(params, ring_sizes):
    """Negative score for optimization."""
    return -_score_ring_config(params, ring_sizes)

def _local_optimize_free(points, time_limit=10.0):
    """Local optimization of all points freely using Powell's method."""
    start_time = time.time()
    
    def obj(x):
        if time.time() > start_time + time_limit:
            return 1e6  # Penalize timeout
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    x0 = points.flatten()
    
    try:
        res = minimize(
            obj, x0, method='Powell',
            options={
                'maxiter': 80, 
                'xtol': 1e-9, 
                'ftol': 1e-11
            }
        )
        
        if time.time() < start_time + time_limit:
            final_score = -res.fun
            if final_score > 0:
                return final_score, res.x.reshape(-1, 2).copy()
                
    except Exception:
        pass
    
    return None

def _refine_with_sa(points, time_limit=15.0, n_restarts=3):
    """
    Refine points using simulated annealing with multiple restarts.
    Returns the best points found.
    """
    start_time = time.time()
    best_points = points.copy()
    best_score = _score(best_points)
    
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > start_time + time_limit - 2:
            break
        
        current_points = best_points.copy()
        current_score = best_score
        
        T_start = 0.005
        T_end = 1e-7
        n_steps = 800
        step_scale = 0.015
        
        for i in range(n_steps):
            if time.time() > start_time + time_limit - 2:
                break
            
            noise = rng.normal(0, step_scale, size=(N, 2))
            new_points = current_points + noise
            
            new_score = _score(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                delta = current_score - new_score
                if delta > 0:
                    T = T_start * (T_end / T_start) ** (i / n_steps)
                    if T > 1e-10:
                        prob = np.exp(-delta / T)
                        if rng.random() < prob:
                            current_points = new_points
                            current_score = new_score
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Define several ring configurations to explore
    # Configuration A: 7 outer + 7 inner (two heptagons)
    ring_sizes_A = [7, 7]
    # Configuration B: 5 outer + 4 middle + 5 inner
    ring_sizes_B = [5, 4, 5]
    # Configuration C: 6 outer + 8 inner
    ring_sizes_C = [6, 8]
    # Configuration D: 4 outer + 5 middle + 5 inner
    ring_sizes_D = [4, 5, 5]
    # Configuration E: 7 outer + 3 middle + 4 inner
    ring_sizes_E = [7, 3, 4]
    
    ring_configs = [
        (ring_sizes_A, [0.7, 0.5], [0.0, np.pi/7]),
        (ring_sizes_B, [0.8, 0.5, 0.2], [0.0, np.pi/5, np.pi/4]),
        (ring_sizes_C, [0.75, 0.4], [0.0, np.pi/6]),
        (ring_sizes_D, [0.85, 0.5, 0.2], [0.0, np.pi/4, np.pi/5]),
        (ring_sizes_E, [0.75, 0.45, 0.15], [0.0, np.pi/7, np.pi/3]),
    ]
    
    # Phase 1: Evaluate initial configurations
    for ring_sizes, radii_init, offsets_init in ring_configs:
        if time.time() > deadline - 60:
            break
        params_init = np.concatenate([radii_init, offsets_init])
        pts = _param_to_points(params_init, ring_sizes)
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Phase 2: Differential evolution on ring parameter spaces
    for ring_sizes, radii_init, offsets_init in ring_configs:
        if time.time() > deadline - 40:
            break
        
        k = len(ring_sizes)
        bounds = [(0.05, 1.0)] * k + [(0, 2 * np.pi / min(ring_sizes))] * k
        
        try:
            result = differential_evolution(
                lambda p: _neg_score_ring_config(p, ring_sizes),
                bounds=bounds,
                seed=42,
                maxiter=30,
                popsize=10,
                tol=1e-6,
                polish=False,
                init='sobol',
                workers=1
            )
            
            if time.time() < deadline - 20:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_points = _param_to_points(result.x, ring_sizes).copy()
                    
        except Exception:
            pass
    
    # Phase 3: Local optimization of best ring configuration
    if best_points is not None and time.time() < deadline - 25:
        result = _local_optimize_free(best_points, time_limit=15.0)
        if result is not None:
            score, points = result
            if score > best_score:
                best_score = score
                best_points = points
    
    # Phase 4: Simulated annealing polish
    if best_points is not None and time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        best_points, best_score = _refine_with_sa(best_points, time_limit=remaining_time, n_restarts=2)
    
    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
