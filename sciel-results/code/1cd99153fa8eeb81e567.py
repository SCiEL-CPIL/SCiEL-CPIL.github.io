"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# EVOLVE-BLOCK-START
def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    Vectorized implementation for speed.
    """
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    # We need to check all triples (i, j, k) with i < j < k
    # We can loop over i and j and vectorize over k
    for i in range(N):
        for j in range(i + 1, N):
            if j + 1 >= N:
                continue
            
            # Vectors from i to j and i to k
            # Cross product magnitude for triangle (i, j, k) is 0.5 * |x_j - x_i| * |y_k - y_i| - ...
            # Actually, area = 0.5 * | (x_j - x_i)(y_k - y_i) - (x_k - x_i)(y_j - y_i) |
            
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            
            # k ranges from j+1 to N-1
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
                    
    return min_area

def _score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    try:
        # Check for degeneracy
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
        
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _objective_reduced(params):
    """
    Objective function for the reduced parameter space.
    params: array of shape (26,).
    P0 is fixed at (0,0).
    P1 is fixed at (r1, 0) where r1 > 0.
    P2...P13 are free in R^2.
    
    We maximize score, so we minimize -score.
    """
    # Reconstruct points
    points = np.zeros((N, 2))
    points[0] = [0.0, 0.0]
    
    # r1 is params[0]
    r1 = params[0]
    if r1 < 1e-6:
        return 1e10 # Penalty for degenerate first edge
    points[1] = [r1, 0.0]
    
    # Remaining 12 points (indices 2 to 13)
    # params[1:25] contain x and y coords for P2...P13
    # There are 12 points, so 24 coordinates
    coords = params[1:25].reshape(12, 2)
    points[2:] = coords
    
    # Check for NaNs/Infs
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
        
    score = _score(points)
    
    if score < 1e-6:
        return 1e10
        
    return -score

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    
    Strategy:
    1. Remove translational and rotational invariance by fixing P0=(0,0) and P1=(r,0).
    2. Use Nelder-Mead local optimization with multiple random restarts.
    3. Initialize restarts using perturbations of the regular 14-gon (adjusted for symmetry).
    4. Normalize the best result to unit area before returning.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Generate initial points from regular 14-gon
    # We need to align it such that P0 is at (0,0) and P1 is on the x-axis?
    # Actually, the symmetry reduction assumes P0 and P1 are specific points.
    # In the optimal configuration, the points are likely in convex position.
    # Let's create a base configuration: Regular 14-gon centered at origin.
    # Then translate/rotate so that one vertex is at (0,0) and the next is on the x-axis?
    # Wait, if we fix P0 and P1, we are assuming the optimal solution has a specific edge.
    # Due to symmetry, any edge can be the "first" edge.
    
    # Let's construct a valid initial guess in the reduced space.
    # Regular 14-gon vertices on unit circle.
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Translate so that the first point is at (0,0)?
    # If we just take reg_pts, P0 is (1,0).
    # Let's shift so P0 is (0,0).
    reg_pts_shifted = reg_pts - reg_pts[0]
    
    # Now P0 is (0,0).
    # P1 is (cos(2pi/14)-1, sin(2pi/14)).
    # We need to rotate so P1 is on the x-axis (y=0).
    angle_p1 = np.arctan2(reg_pts_shifted[1, 1], reg_pts_shifted[1, 0])
    R = np.array([[np.cos(-angle_p1), -np.sin(-angle_p1)],
                  [np.sin(-angle_p1), np.cos(-angle_p1)]])
    
    reg_pts_rotated = reg_pts_shifted @ R.T
    
    # Now P0 is (0,0) and P1 is (r, 0) with r > 0.
    # Extract parameters for the reduced space
    r1 = reg_pts_rotated[1, 0]
    coords_rest = reg_pts_rotated[2:]
    
    initial_params = np.concatenate(([r1], coords_rest.flatten()))
    
    best_params = initial_params
    best_score_val = -_objective_reduced(initial_params)
    
    # Multi-start Nelder-Mead
    # We will try a few restarts.
    # Random perturbations of the initial guess.
    
    rng = np.random.default_rng(42)
    
    num_restarts = 5
    max_iter_per_restart = 2000
    
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Generate a perturbed start
        # Scale factor for perturbation
        scale = 0.1 if i > 0 else 0.0
        if i == 0:
            current_params = initial_params.copy()
        else:
            # Perturb coordinates
            noise = rng.normal(0, scale, size=initial_params.shape)
            current_params = initial_params + noise
            # Ensure r1 is positive
            current_params[0] = np.abs(current_params[0]) + 1e-3
        
        try:
            res = minimize(
                _objective_reduced,
                current_params,
                method='Nelder-Mead',
                options={
                    'maxiter': max_iter_per_restart,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if res.fun < _objective_reduced(best_params):
                best_params = res.x
                best_score_val = -res.fun
                
        except Exception:
            continue
            
    # Reconstruct final points from best_params
    points = np.zeros((N, 2))
    points[0] = [0.0, 0.0]
    points[1] = [best_params[0], 0.0]
    points[2:] = best_params[1:25].reshape(12, 2)
    
    # Normalize to unit area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        # Fallback to regular polygon if hull fails
        final_points = reg_pts.copy()
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
        return final_points
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = points * scale
    else:
        final_points = reg_pts.copy()
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale

    return final_points
# EVOLVE-BLOCK-END
