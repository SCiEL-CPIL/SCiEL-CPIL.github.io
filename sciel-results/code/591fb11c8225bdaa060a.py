import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13

# EVOLVE-BLOCK-START

def _compute_triangle_areas(points):
    """
    Compute areas of all triangles formed by triples of points.
    Returns a flattened array of areas for i < j < k.
    """
    n = len(points)
    areas = np.zeros((n * (n - 1) * (n - 2)) // 6)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Vectorized area calculation
                # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                areas[idx] = 0.5 * abs(cross)
                idx += 1
    return areas

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    
    areas = _compute_triangle_areas(points)
    min_area = np.min(areas)
    
    if min_area <= 1e-12:
        return 0.0
    
    return min_area / hull_area

def _objective(x, deadline):
    """
    Objective function for minimization: minimize negative score.
    x is a flat array of 2*N coordinates.
    """
    if time.time() > deadline:
        return 1e10
    
    points = x.reshape(N, 2)
    
    # We don't strictly need to normalize inside the objective if the score is scale-invariant,
    # but normalizing helps keep the search space bounded and prevents numerical issues
    # if the optimizer drifts to very small or very large scales.
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    # Normalize to unit hull area for stable gradient-free search?
    # Actually, the ratio is invariant to scaling. 
    # However, Nelder-Mead works better if the scale is consistent.
    # Let's normalize to have hull area = 1.
    scale = 1.0 / np.sqrt(hull_area)
    center = np.mean(points, axis=0)
    points_norm = (points - center) * scale
    
    score = _score(points_norm)
    if score <= 0:
        return 1e10
    return -score

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def construct():
    start_time = time.time()
    # 120s limit, use 110s for safety
    deadline = start_time + 110.0
    
    best_points = None
    best_score = 0.0
    
    # Initial guess: Regular 13-gon
    # The regular polygon is a strong candidate for the global optimum or a very close local optimum.
    init_points = _regular_polygon(N)
    
    # Evaluate initial score
    init_score = _score(init_points)
    if init_score > best_score:
        best_score = init_score
        best_points = init_points.copy()
    
    # Optimization Strategy:
    # 1. Local search from the regular polygon using Nelder-Mead.
    # 2. If time permits, try a few random perturbations of the regular polygon and re-optimize.
    
    def run_optimization(initial_flat, max_iter):
        try:
            result = minimize(
                _objective,
                initial_flat,
                args=(deadline,),
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            return result
        except Exception:
            return None

    # Run 1: From regular polygon
    res = run_optimization(init_points.flatten(), 500)
    if res is not None and res.fun < -best_score:
        new_points = res.x.reshape(N, 2)
        # Normalize for return
        hull_area = _hull_area(new_points)
        if hull_area > 1e-12:
            scale = 1.0 / np.sqrt(hull_area)
            center = np.mean(new_points, axis=0)
            new_points_norm = (new_points - center) * scale
            new_score = _score(new_points_norm)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points_norm.copy()
    
    # Run 2: Perturbed regular polygon
    # Add small random noise to break symmetry and potentially find a better local minimum
    rng = np.random.default_rng(42)
    for i in range(3):
        if time.time() > deadline - 10:
            break
        
        noise = rng.normal(0, 0.05, size=(N, 2))
        perturbed_points = init_points + noise
        perturbed_flat = perturbed_points.flatten()
        
        res = run_optimization(perturbed_flat, 300)
        if res is not None and res.fun < -best_score:
            new_points = res.x.reshape(N, 2)
            hull_area = _hull_area(new_points)
            if hull_area > 1e-12:
                scale = 1.0 / np.sqrt(hull_area)
                center = np.mean(new_points, axis=0)
                new_points_norm = (new_points - center) * scale
                new_score = _score(new_points_norm)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points_norm.copy()

    # If we still have time, do a very fine-tuned search on the best point found
    if best_points is not None and time.time() < deadline - 5:
        best_flat = best_points.flatten()
        res = run_optimization(best_flat, 1000)
        if res is not None and res.fun < -best_score:
            new_points = res.x.reshape(N, 2)
            hull_area = _hull_area(new_points)
            if hull_area > 1e-12:
                scale = 1.0 / np.sqrt(hull_area)
                center = np.mean(new_points, axis=0)
                new_points_norm = (new_points - center) * scale
                new_score = _score(new_points_norm)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points_norm.copy()

    if best_points is None:
        return _regular_polygon(N)
        
    return best_points

# EVOLVE-BLOCK-END
