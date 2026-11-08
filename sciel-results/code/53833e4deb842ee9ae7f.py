import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

def _compute_triangle_areas(points):
    """
    Compute areas of all triangles formed by N points.
    Returns array of shape (C(N,3),)
    """
    n = points.shape[0]
    num_triangles = n * (n - 1) * (n - 2) // 6
    areas = np.zeros(num_triangles)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |det(p_j - p_i, p_k - p_i)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                det = v1[0] * v2[1] - v1[1] * v2[0]
                areas[idx] = 0.5 * np.abs(det)
                idx += 1
    return areas

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return -1e10
    min_area = np.min(_compute_triangle_areas(points))
    return min_area / hull_area

def _params_to_points(params):
    """
    params: shape (N, 2) where col 0 is angle, col 1 is radius.
    Returns points in Cartesian coordinates.
    """
    angles = params[:, 0]
    radii = params[:, 1]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _objective(params):
    """
    Objective to minimize: - (soft_min_area / hull_area)
    We use a soft minimum to make the objective differentiable.
    """
    points = _params_to_points(params)
    
    # Ensure points are well-behaved
    if not np.all(np.isfinite(points)):
        return 1e10
        
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-8:
        return 1e10

    areas = _compute_triangle_areas(points)
    
    # Soft minimum: approximates min(areas) but is smooth
    # alpha controls the "sharpness". Higher alpha -> closer to true min.
    # We adapt alpha or use a fixed reasonable value.
    alpha = 200.0 
    soft_min = -np.log(np.sum(np.exp(-alpha * areas))) / alpha
    
    score = soft_min / hull_area
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin for 150s limit
    
    best_score = -np.inf
    best_params = None
    
    # Initial guesses in (angle, radius) space
    # 1. Regular polygon on unit circle
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    base_radii = np.ones(N)
    base_params = np.column_stack([base_angles, base_radii])
    
    initial_guesses = [base_params]
    
    # 2. Perturbed regular polygon
    rng = np.random.default_rng(42)
    for _ in range(5):
        noise_angles = rng.normal(0, 0.05, N)
        noise_radii = rng.normal(0, 0.05, N)
        # Keep radii positive
        new_radii = np.maximum(0.5, base_radii + noise_radii)
        new_angles = base_angles + noise_angles
        initial_guesses.append(np.column_stack([new_angles, new_radii]))
        
    # 3. Points on a square? (Might be a different local minimum)
    # Just in case, try a few random convex configurations
    for _ in range(3):
        # Random points in unit disk, then project to hull? 
        # Actually, for max min area, points should be on hull.
        # Let's just randomize angles and radii slightly
        random_angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        random_radii = rng.uniform(0.8, 1.0, N)
        initial_guesses.append(np.column_stack([random_angles, random_radii]))

    # Optimization settings
    # L-BFGS-B is good for box-constrained problems.
    # Bounds: Angles [0, 2pi], Radii [0.1, 2.0] (allowing some expansion)
    bounds = [(0, 2 * np.pi), (0.1, 2.0)] * N
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
            
        try:
            # L-BFGS-B with numerical gradients (since analytical is complex for soft-min + hull area)
            # However, numerical gradients on 28 dimensions are fine.
            res = minimize(
                _objective, 
                guess.flatten(), 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if time.time() > deadline:
                break
                
            if res.success or res.status in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                # Evaluate the actual (non-soft) score of the result
                final_points = _params_to_points(res.x.reshape(-1, 2))
                actual_score = _score(final_points)
                
                if actual_score > best_score:
                    best_score = actual_score
                    best_params = res.x.reshape(-1, 2)
                    
        except Exception:
            pass

    # If L-BFGS-B failed or didn't find anything, try Nelder-Mead on the constrained space
    # as a fallback, or just return the best found.
    # Given the differentiability, L-BFGS-B should be robust.
    
    if best_params is None:
        # Fallback to regular polygon
        return _params_to_points(base_params)
        
    final_points = _params_to_points(best_params)
    
    # Final polish with Nelder-Mead on Cartesian coordinates if time permits
    # This can help escape the "soft-min" approximation error
    if time.time() < deadline - 20:
        try:
            x0 = final_points.flatten()
            res_nm = minimize(
                lambda x: -_score(x.reshape(-1, 2)),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            score_nm = -res_nm.fun
            if score_nm > best_score:
                final_points = res_nm.x.reshape(-1, 2)
                best_score = score_nm
        except Exception:
            pass
            
    return final_points

if __name__ == "__main__":
    pts = construct()
    print(_score(pts))
