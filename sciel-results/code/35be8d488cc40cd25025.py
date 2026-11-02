import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145

# Precompute triplet indices
_triplet_indices = list(combinations(range(N), 3))
_n_triplets = len(_triplet_indices)
_idx_arr = np.array(_triplet_indices)  # shape (165, 3)

def _min_area(points):
    """Fast min area using precomputed indices."""
    p_i = points[_idx_arr[:, 0]]
    p_j = points[_idx_arr[:, 1]]
    p_k = points[_idx_arr[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return 0.5 * np.min(cross)

def _project_to_triangle(points):
    """Project points to be inside the equilateral triangle."""
    points = points.copy()
    for i in range(len(points)):
        x, y = points[i]
        # Clip y to [0, S3H]
        y = max(0.0, min(S3H, y))
        # Clip x based on y
        x_min = y / np.sqrt(3)
        x_max = 1.0 - y / np.sqrt(3)
        x = max(x_min, min(x_max, x))
        points[i] = [x, y]
    return points

def _nelder_mead_obj(points_flat):
    """Objective for Nelder-Mead: negative min area."""
    points = points_flat.reshape(-1, 2)
    points = _project_to_triangle(points)
    return -_min_area(points)

def _nelder_mead_obj_with_proj(points_flat):
    """Objective that projects back and returns the projected points implicitly."""
    points = points_flat.reshape(-1, 2)
    points = _project_to_triangle(points)
    return -_min_area(points)

def construct():
    start_time = time.time()
    
    # Generate initial points using barycentric coordinates for validity
    # Use a grid-like initial guess to avoid degenerate configurations
    # Vertices: (0,0), (1,0), (0.5, S3H)
    # We'll use a stratified sampling approach
    
    def _bary_to_cart(bary):
        # bary shape (N, 3), sum to 1
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 0.0])
        v3 = np.array([0.5, S3H])
        return bary[:, 0:1] * v1 + bary[:, 1:2] * v2 + bary[:, 2:3] * v3

    def _random_bary(n, rng):
        u = rng.random((n, 2))
        u.sort(axis=1)
        # Sort to get stratified samples in [0,1]
        # Convert to barycentric: (1-u2, u2-u1, u1) where u1<u2
        bary = np.zeros((n, 3))
        bary[:, 0] = 1.0 - u[:, 1]
        bary[:, 1] = u[:, 1] - u[:, 0]
        bary[:, 2] = u[:, 0]
        return bary

    rng = np.random.default_rng(42)
    
    best_points = None
    best_val = float('inf')
    
    # Multi-start Nelder-Mead
    n_restarts = 50
    time_per_restart = (DEADLINE - 10) / n_restarts
    
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
        
        # Generate initial points
        if restart == 0:
            # Symmetric initial guess
            # Place points on a grid inside the triangle
            # Use barycentric grid
            # For simplicity, use random stratified
            bary = _random_bary(N, rng)
        else:
            bary = _random_bary(N, rng)
        
        points = _bary_to_cart(bary)
        points = _project_to_triangle(points)
        
        # Run Nelder-Mead
        try:
            result = minimize(
                _nelder_mead_obj,
                points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if result.success or result.fun < best_val:
                new_points = result.x.reshape(-1, 2)
                new_points = _project_to_triangle(new_points)
                val = -result.fun
                if val > best_val:
                    best_val = val
                    best_points = new_points.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback
        bary = _random_bary(N, rng)
        best_points = _bary_to_cart(bary)
        best_points = _project_to_triangle(best_points)
    
    # Final polish with a tighter tolerance if time allows
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 2:
        try:
            result = minimize(
                _nelder_mead_obj,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < -best_val:
                new_points = result.x.reshape(-1, 2)
                new_points = _project_to_triangle(new_points)
                best_points = new_points
        except Exception:
            pass
    
    return best_points
