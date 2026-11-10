import numpy as np
import time
import math

def _build_grid():
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X.ravel(), Y.ravel()

def _build_subgrid(step=3):
    xs = np.arange(-3.0, 1.5, 0.01 * step)
    ys = np.arange(-1.5, 1.5, 0.01 * step)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X.ravel(), Y.ravel()

def _densify_path(path):
    """Densify path by adding linear interpolations between consecutive waypoints."""
    full = np.vstack([[0.0, 0.0, 0.0], path, [-2.0, 0.0, 90.0]])
    result = [full[0]]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(math.ceil(da)))
        for k in range(1, m + 1):
            t = k / m
            interp = p1 + t * (p2 - p1)
            result.append(interp)
    return np.array(result)

def _evaluate_path_fast(path, grid_x, grid_y, mask=None):
    """Evaluate the area of the sofa (intersection of all placements)."""
    dense = _densify_path(path)
    n_points = len(grid_x)
    if mask is None:
        inside = np.ones(n_points, dtype=bool)
    else:
        inside = mask.copy()
    
    for i in range(len(dense)):
        x, y, a_deg = dense[i]
        a_rad = math.radians(a_deg)
        cos_a = math.cos(a_rad)
        sin_a = math.sin(a_rad)
        
        dx = grid_x - x
        dy = grid_y - y
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        in_base = arm1 | arm2
        
        inside = inside & in_base
        if not np.any(inside):
            break
    
    return np.sum(inside) * 0.0001

def _coordinate_descent(path, grid_x, grid_y, deadline, n_iter=500):
    """Coordinate descent with adaptive step sizes."""
    n = len(path)
    best_path = path.copy()
    full_mask = np.ones(len(grid_x), dtype=bool)
    best_area = _evaluate_path_fast(best_path, grid_x, grid_y, full_mask)
    
    step_sizes = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for ss in step_sizes:
        for _ in range(n_iter):
            if time.time() > deadline:
                return best_path, best_area
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    return best_path, best_area
                for dim in range(3):
                    for direction in [1, -1]:
                        trial = best_path.copy()
                        trial[i, dim] += direction * ss
                        if dim == 2:
                            trial[i, dim] = np.clip(trial[i, dim], -10, 100)
                        
                        area = _evaluate_path_fast(trial, grid_x, grid_y, full_mask)
                        if area > best_area:
                            best_path = trial
                            best_area = area
                            improved = True
            if not improved:
                break
    
    return best_path, best_area

def _ps_path(n_points=20):
    """Generate the Pogossnitsky-Scheybman analytic optimal path.
    
    The known near-optimal path for moving a unit-width sofa through a unit-width
    L-corridor uses the parametric curve:
    x = -1 + cos(theta) - sin(theta)
    y = -1 + sin(theta) + cos(theta)
    a = theta (in radians)
    
    But we need to map this to our coordinate system where:
    - Start: (0, 0, 0)
    - End: (-2, 0, 90)
    
    The PS path in the standard formulation has the inner corner at the origin
    and the sofa moves from one arm to the other. We need to adapt this.
    """
    # The PS path for the standard L-corridor problem
    # The sofa's inner corner traces a specific curve
    theta = np.linspace(0, math.pi/2, n_points)
    
    # Standard PS path (for corridor with inner corner at origin)
    # x_corner = cos(t) - sin(t)
    # y_corner = sin(t) + cos(t)
    # But we need to shift to match our start/end points
    
    # At t=0: x = 1, y = 1, a = 0
    # At t=pi/2: x = -1, y = 1, a = 90
    
    # We need to map: start (0,0,0) -> end (-2,0,90)
    # The PS path gives: start (1,1,0) -> end (-1,1,90)
    # So we shift: x -> x - 1, y -> y - 1
    # That gives: start (0,0,0) -> end (-2,0,90) ✓
    
    x = np.cos(theta) - np.sin(theta) - 1.0
    y = np.sin(theta) + np.cos(theta) - 1.0
    angles = np.degrees(theta)
    
    return np.column_stack([x, y, angles])

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600  # 30s safety margin before 630s limit
    
    # Build grids
    sub_x, sub_y = _build_subgrid(step=3)
    
    # Start with the Pogossnitsky-Scheybman analytic path
    n_ctrl = 20
    best_params = _ps_path(n_ctrl)
    
    # Evaluate on subgrid
    best_area_sub = _evaluate_path_fast(best_params, sub_x, sub_y, np.ones(len(sub_x), dtype=bool))
    
    # Coordinate descent on subgrid
    best_params, best_area_sub = _coordinate_descent(best_params, sub_x, sub_y, deadline, n_iter=300)
    
    if time.time() > deadline:
        return best_params
    
    # Try variations with different numbers of control points
    best_overall = best_params.copy()
    best_overall_area = best_area_sub
    
    for n_pts in [10, 15, 25, 30]:
        if time.time() > deadline:
            break
        alt = _ps_path(n_pts)
        area_alt = _evaluate_path_fast(alt, sub_x, sub_y, np.ones(len(sub_x), dtype=bool))
        if area_alt > best_overall_area:
            best_overall = alt.copy()
            best_overall_area = area_alt
    
    # Final coordinate descent on best path
    best_params, best_area_sub = _coordinate_descent(best_overall, sub_x, sub_y, deadline, n_iter=200)
    
    if time.time() > deadline:
        return best_params
    
    # Validate on full grid
    full_x, full_y = _build_grid()
    full_area = _evaluate_path_fast(best_params, full_x, full_y, np.ones(len(full_x), dtype=bool))
    
    # If there's time, do a final refinement on full grid
    if time.time() < deadline - 60:
        best_params, full_area = _coordinate_descent(best_params, full_x, full_y, deadline, n_iter=100)
    
    return best_params

# EVOLVE-BLOCK-END
