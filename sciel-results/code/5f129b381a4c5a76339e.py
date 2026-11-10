# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def moving_sofa_2d() -> np.ndarray:
    """
    The 2D moving-sofa problem: find the largest-area rigid shape that can be moved
    through an L-shaped corridor of width 1 that makes a single 90-degree turn.

    The shape is specified IMPLICITLY by a motion "path". Each path point is a triple
    (x, y, angle_degrees) giving the position of the corridor's inner corner and the
    corridor's rotation at one instant of the move. In the reference frame of the sofa
    the corridor sweeps over the plane; the sofa is then the intersection of every
    corridor placement along the path, so by construction it always clears the corridor.
    The objective is to maximize the area of that intersection.

    The evaluator prepends the start state (0, 0, 0) and appends the end state
    (-2, 0, 90), then densifies the path to <=1-degree rotation steps, exactly as in the
    AlphaEvolve "single turn" corridor notebook.

    Returns:
        path: np.ndarray of shape (n, 3); each row is (x, y, angle_degrees) for the
            corridor's inner corner. Here n = 8. Endpoints are added by the evaluator.
    """
    
    # Grid points for the sofa
    # i = 0..449, j = 0..299
    # x_grid = -3 + 0.01 * i
    # y_grid = -1.5 + 0.01 * j
    i_range = np.arange(-3, 1.5, 0.01)
    j_range = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(i_range, j_range, indexing='ij')
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    n_points = len(grid_points)

    # Start and end states added by evaluator
    start_state = np.array([0.0, 0.0, 0.0])
    end_state = np.array([-2.0, 0.0, 90.0])

    def evaluate_path(path):
        """
        Calculate the number of grid points inside the intersection of all placements.
        path: np.array of shape (n, 3)
        """
        # Prepend start, append end
        full_path = np.vstack([start_state, path, end_state])
        
        # Densify path
        placements = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i + 1]
            da = p2[2] - p1[2]
            m = max(1, int(math.ceil(abs(da))))
            # Interpolate in m steps
            # We want m+1 points including start and end of this segment?
            # "adds linear interpolations of (x, y, a) in m = max(1, ceil(|a2 - a1|)) equal steps"
            # This usually means m intervals, so m+1 points. But the endpoints are shared.
            # Let's generate t from 1 to m (excluding start which is already added, including end)
            for k in range(1, m + 1):
                t = k / m
                p = p1 + t * (p2 - p1)
                placements.append(p)
        
        # Check intersection
        # A point is in the sofa if it is in ALL placements
        in_sofa = np.ones(n_points, dtype=bool)
        
        for p in placements:
            x, y, a_deg = p
            a_rad = np.deg2rad(a_deg)
            
            # Rotate grid points by -a around (x, y)
            # w' = R(-a) (w - (x, y))
            # R(-a) = [[cos(a), sin(a)], [-sin(a), cos(a)]]
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            # Translate
            dx = grid_points[:, 0] - x
            dy = grid_points[:, 1] - y
            
            # Rotate
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            # Base corridor:
            # Union of {0 <= u <= 1, -3 <= v <= 1} and {-3 <= u <= 1, 0 <= v <= 1}
            # Let's check membership in the union
            # Rect 1: 0 <= u <= 1 AND -3 <= v <= 1
            # Rect 2: -3 <= u <= 1 AND 0 <= v <= 1
            
            in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            in_corridor = in_rect1 | in_rect2
            
            in_sofa &= in_corridor
            
            # Early exit if empty
            if not np.any(in_sofa):
                return 0
        
        return np.sum(in_sofa)

    # Initial path from seed
    best_path = np.array(
        [
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ],
        dtype=np.float64,
    )
    
    best_score = evaluate_path(best_path)
    
    # Optimization loop
    # We will use a simple local search with random restarts.
    # Time limit: 600s. We should stop around 580s to be safe.
    start_time = time.time()
    time_limit = 580.0
    
    # Parameters for search
    # We perturb x, y, a
    # x, y ranges: roughly -1 to 1
    # a ranges: 0 to 90, but must be monotonic? The problem doesn't strictly require monotonicity, 
    # but non-monotonic might be inefficient or invalid if it causes backtracking issues? 
    # Actually, the path just defines placements. Backtracking is allowed physically if the sofa fits, 
    # but it doesn't help maximize the intersection (it only shrinks it). So we assume monotonic a is optimal.
    # Let's enforce monotonicity of a in the search to reduce search space and ensure validity.
    
    def enforce_monotonic_a(path):
        # Ensure a is strictly increasing
        for i in range(1, len(path)):
            if path[i, 2] <= path[i-1, 2]:
                path[i, 2] = path[i-1, 2] + 0.1
        # Ensure within 0..90
        path[:, 2] = np.clip(path[:, 2], 0.1, 89.9)
        return path

    # Simple Nelder-Mead like or Coordinate Descent
    # Let's use Coordinate Descent with adaptive step size
    
    current_path = best_path.copy()
    current_score = best_score
    
    # Restart logic
    max_restarts = 10
    restarts = 0
    
    while time.time() - start_time < time_limit:
        if restarts > 0:
            # Perturb current best significantly for a new restart
            # Or just start from a random valid path?
            # Let's try local search around best_path with random perturbations
            current_path = best_path + np.random.normal(0, 0.2, best_path.shape)
            current_path = enforce_monotonic_a(current_path)
            current_score = evaluate_path(current_path)
            if current_score > best_score:
                best_score = current_score
                best_path = current_path.copy()
        
        # Local search
        step_size = 0.05
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Try perturbing each dimension
            for i in range(len(current_path)):
                for j in range(3):
                    # Try positive and negative steps
                    for delta in [step_size, -step_size]:
                        trial_path = current_path.copy()
                        trial_path[i, j] += delta
                        
                        # Ensure a is monotonic and within bounds
                        if j == 2:
                            if trial_path[i, 2] < 0 or trial_path[i, 2] > 90:
                                continue
                            # Check monotonicity
                            if i > 0 and trial_path[i, 2] <= trial_path[i-1, 2]:
                                continue
                            if i < len(trial_path)-1 and trial_path[i, 2] >= trial_path[i+1, 2]:
                                continue
                        else:
                            # x, y bounds
                            if trial_path[i, j] < -2 or trial_path[i, j] > 2:
                                continue
                        
                        trial_score = evaluate_path(trial_path)
                        if trial_score > current_score:
                            current_path = trial_path
                            current_score = trial_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_path = current_path.copy()
                            # Break to restart dimension loop with new center
                            break
                    if improved:
                        break
                if improved:
                    break
            
            # If no improvement, reduce step size
            if not improved:
                step_size *= 0.5
                if step_size < 0.001:
                    break
        
        restarts += 1

    return best_path

# EVOLVE-BLOCK-END
