# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_grid():
    """Precompute grid points for sofa membership testing."""
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X.ravel(), Y.ravel()


def _densify_path(path):
    """
    Densify a path the same way the evaluator does.
    Prepends (0,0,0) and appends (-2,0,90).
    Between consecutive rows, adds linear interpolations in m = max(1, ceil(|a2-a1|)) steps.
    """
    # Prepend start, append end
    full = np.vstack([[0.0, 0.0, 0.0], path, [-2.0, 0.0, 90.0]])
    
    # Densify
    result = [full[0]]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        for k in range(1, m + 1):
            t = k / m
            interp = p1 + t * (p2 - p1)
            result.append(interp)
    return np.array(result)


def _check_in_base(u, v):
    """
    Check if point (u, v) is in the base L-shaped corridor.
    Base: {0 <= u <= 1, -3 <= v <= 1} ∪ {-3 <= u <= 1, 0 <= v <= 1}
    Returns boolean array.
    """
    # Arm 1: 0 <= u <= 1, -3 <= v <= 1
    arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    # Arm 2: -3 <= u <= 1, 0 <= v <= 1
    arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    return arm1 | arm2


def _evaluate_path(path, grid_x, grid_y):
    """
    Evaluate the sofa area for a given path.
    Returns the raw value (area).
    """
    densified = _densify_path(path)
    n_points = len(grid_x)
    
    # Start with all points as potential sofa members
    inside_all = np.ones(n_points, dtype=bool)
    
    for (cx, cy, a_deg) in densified:
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # R(-a) * (w - c)
        # w - c = (grid_x - cx, grid_y - cy)
        dx = grid_x - cx
        dy = grid_y - cy
        
        # R(-a) = [[cos(a), sin(a)], [-sin(a), cos(a)]]
        # So R(-a)(dx, dy) = (dx*cos(a) + dy*sin(a), -dx*sin(a) + dy*cos(a))
        u = dx * cos_a + dy * sin_a
        v = -dx * sin_a + dy * cos_a
        
        in_base = _check_in_base(u, v)
        inside_all &= in_base
        
        # Early exit if no points remain
        if not np.any(inside_all):
            return 0.0
    
    return float(np.sum(inside_all)) * 0.0001


def moving_sofa_2d() -> np.ndarray:
    """
    Optimize the moving sofa path to maximize the sofa area.
    Uses a time-bounded local search to avoid timeouts.
    """
    start_time = time.time()
    time_limit = 580  # Leave margin for safety (limit is 630)
    
    # Build grid
    grid_x, grid_y = _build_grid()
    
    # Initial path: A heuristic path for the sofa corner problem.
    # The sofa needs to rotate 90 degrees. The critical part is the middle rotation.
    # A common optimal-ish shape is an arc.
    # Let's try a path that keeps the inner corner of the L-shape moving along a circle
    # or close to the wall corner.
    
    # The base L-shape inner corner is at (0,0) relative to the placement center?
    # No, the placement is defined by "inner corner at (x,y)".
    # Base corridor: {0 <= u <= 1, -3 <= v <= 1} U {-3 <= u <= 1, 0 <= v <= 1}.
    # The "inner corner" of the L-shape (the reflex vertex) is at (0,0) in local coords.
    # So (x,y) is the position of this reflex vertex.
    
    # To move the sofa, the reflex vertex should stay close to the wall corner (0,0) of the corridor?
    # The corridor is fixed? No, the "corridor" is the base shape. The "sofa" is the grid points.
    # Wait, re-reading: "Row (x, y, a) places the width-1 L-shaped corridor...".
    # "The sofa is the set of grid points ... that are inside every placement".
    # So the "sofa" is the intersection of all placements of the L-shape.
    # We want to maximize the area of this intersection.
    
    # The start is (0,0,0). The end is (-2,0,90).
    # At a=0, the L-shape is at (0,0).
    # At a=90, the L-shape is at (-2,0).
    
    # A good strategy is to move the vertex (x,y) in a way that maximizes the overlap.
    # For a=0, the L occupies x in [0,1] (arm1) and y in [-3,1].
    # For a=90, rotated 90 deg CCW around (x,y).
    # The base L has arms extending from (0,0) to (1,0) and (0,1) roughly?
    # Arm 1: u in [0,1], v in [-3,1]. This is a vertical strip of width 1, height 4, shifted down.
    # Arm 2: u in [-3,1], v in [0,1]. This is a horizontal strip of width 4, height 1, shifted left.
    # The overlap (the corner) is [0,1]x[0,1].
    
    # Let's use a simple local search with coordinate perturbation.
    
    # Start with a reasonable guess.
    # The vertex needs to move from (0,0) to (-2,0) while rotating 0 to 90.
    # A linear interpolation of position and angle might be a start, but usually,
    # the vertex needs to move "out" of the corner to allow rotation.
    
    best_path = np.array([
        [0.0, 0.0, 10.0],
        [-0.2, 0.2, 20.0],
        [-0.5, 0.5, 30.0],
        [-0.8, 0.8, 45.0],
        [-1.2, 0.8, 60.0],
        [-1.6, 0.5, 75.0],
        [-1.9, 0.2, 85.0]
    ], dtype=np.float64)
    
    best_score = _evaluate_path(best_path, grid_x, grid_y)
    
    # Simple Local Search: Coordinate Descent with random restarts if time permits.
    # Since DE was too slow, we will do a few iterations of coordinate perturbation.
    
    n_params = best_path.size
    bounds = [(-2.0, 1.0), (-1.0, 1.0), (0.0, 90.0)] * 7 # 7 points
    
    def objective(params):
        path = params.reshape(7, 3)
        # Ensure angles are monotonically increasing
        for i in range(1, 7):
            if path[i, 2] < path[i-1, 2]:
                path[i, 2] = path[i-1, 2]
        return -_evaluate_path(path, grid_x, grid_y)
    
    current_params = best_path.ravel().copy()
    current_score = -best_score
    
    # Define step sizes
    step_sizes = [0.1, 0.1, 0.5] * 7
    
    # Iterative improvement
    max_iterations = 5000 # Cap iterations to prevent infinite loop, time guard is primary
    for i_iter in range(max_iterations):
        if time.time() - start_time > time_limit - 10:
            break
        
        improved = False
        for i in range(n_params):
            # Try positive and negative steps
            for sign in [1, -1]:
                new_params = current_params.copy()
                new_params[i] += sign * step_sizes[i]
                
                # Apply bounds
                b_min, b_max = bounds[i]
                if new_params[i] < b_min: new_params[i] = b_min
                if new_params[i] > b_max: new_params[i] = b_max
                
                # Check monotonicity constraint for angles
                path = new_params.reshape(7, 3)
                valid = True
                for j in range(1, 7):
                    if path[j, 2] < path[j-1, 2]:
                        # If invalid, snap to previous
                        path[j, 2] = path[j-1, 2]
                        # This might make the move less effective, but keeps it valid
                        # Actually, if we snap, the parameter change might be nullified.
                        # Let's just evaluate.
                
                new_score = objective(new_params)
                
                if new_score < current_score:
                    current_params = new_params
                    current_score = new_score
                    best_score = -current_score
                    best_path = path.copy() # Note: path is modified in objective? No, objective takes params.
                    # Re-construct best_path from current_params to be safe
                    best_path = current_params.reshape(7, 3)
                    # Re-apply monotonicity to best_path for consistency?
                    # The evaluator handles interpolation, but the path itself should be sane.
                    for j in range(1, 7):
                        if best_path[j, 2] < best_path[j-1, 2]:
                            best_path[j, 2] = best_path[j-1, 2]
                            
                    improved = True
                    # Reduce step size slightly on improvement? Or keep constant for simplicity.
                    # Let's keep constant for now to allow larger jumps.
                    break # Restart loop for this dimension? Or continue?
                    # Continuing is fine, but breaking out of sign loop is good.
            
            # If we improved, we might want to reset step size or continue.
            # For simplicity, we just continue to the next dimension.
            
            # Time check inside inner loop to be safe
            if time.time() - start_time > time_limit - 10:
                break
        
        if not improved:
            # Try a random perturbation if stuck?
            # Given time constraints, simple coordinate descent might get stuck in local minima.
            # However, the initial guess is likely close to a good basin.
            # If we want to be safer, we can add a few random restarts if time > 200s remaining.
            if time.time() - start_time < time_limit - 100:
                # Random restart in the vicinity
                current_params = best_path.ravel().copy() + np.random.normal(0, 0.1, n_params)
                # Clip to bounds
                for i in range(n_params):
                    b_min, b_max = bounds[i]
                    current_params[i] = np.clip(current_params[i], b_min, b_max)
                # Enforce monotonicity on angles
                path = current_params.reshape(7, 3)
                for j in range(1, 7):
                    if path[j, 2] < path[j-1, 2]:
                        path[j, 2] = path[j-1, 2]
                current_params = path.ravel()
                current_score = objective(current_params)
                if -current_score > best_score:
                    best_score = -current_score
                    best_path = path.copy()
            else:
                break # No time for restarts, and no local improvement
    
    return best_path

# EVOLVE-BLOCK-END
