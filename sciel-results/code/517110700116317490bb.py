# EVOLVE-BLOCK-START
import numpy as np
import time


def moving_sofa_2d() -> np.ndarray:
    """
    The 2D moving-sofa problem: find the largest-area rigid shape that can be moved
    through an L-shaped corridor of width 1 that makes a single 90-degree turn.
    """
    start_time = time.time()
    deadline = 620.0  # evaluator kills at 630s, keep safety margin

    # Sofa grid
    x_grid = np.arange(-3, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid)
    P = np.column_stack((X.ravel(), Y.ravel()))  # (N, 2)
    N = P.shape[0]
    Px = P[:, 0]
    Py = P[:, 1]

    # Start and end states (added by evaluator)
    start_state = np.array([0.0, 0.0, 0.0])
    end_state = np.array([-2.0, 0.0, 90.0])

    def densify_path(path):
        """
        Densify the path by adding linear interpolations between consecutive rows.
        Between consecutive rows (x1,y1,a1) and (x2,y2,a2), add m = max(1, ceil(|a2-a1|))
        equal steps of interpolation.
        """
        if len(path) == 0:
            return np.array([start_state, end_state])
        
        full_path = [start_state] + list(path) + [end_state]
        
        densified = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(1, m + 1):
                t = k / m
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                densified.append([x, y, a])
        
        return np.array(densified)

    def evaluate_path_fast(path):
        """
        Fast evaluation of the sofa area for a given path.
        Uses vectorized operations and early termination.
        """
        densified = densify_path(path)
        n_placements = len(densified)
        
        in_sofa = np.ones(N, dtype=bool)
        
        for idx in range(n_placements):
            x, y, a_deg = densified[idx]
            a_rad = np.radians(a_deg)
            dx = Px - x
            dy = Py - y
            cos_a = np.cos(-a_rad)
            sin_a = np.sin(-a_rad)
            u = cos_a * dx - sin_a * dy
            v = sin_a * dx + cos_a * dy
            
            eps = 1e-9
            in_r1 = (u >= -eps) & (u <= 1 + eps) & (v >= -3 - eps) & (v <= 1 + eps)
            in_r2 = (u >= -3 - eps) & (u <= 1 + eps) & (v >= -eps) & (v <= 1 + eps)
            
            mask = in_r1 | in_r2
            in_sofa &= mask
            
            # Early termination
            count = in_sofa.sum()
            if count == 0:
                return 0.0
            
            # Periodic check for speed (every 10 placements)
            # Actually, let's just check every time since it's cheap
        
        return in_sofa.sum() * 0.0001

    # Initial path - hand-tuned starting point
    path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)

    best_path = path.copy()
    best_score = evaluate_path_fast(path)

    def coordinate_descent_optimize(path, max_iterations=200, base_step=0.1):
        """
        Coordinate descent optimization with adaptive step sizes.
        """
        current_path = path.copy()
        current_score = evaluate_path_fast(current_path)
        
        for iteration in range(max_iterations):
            if (time.time() - start_time) > deadline - 30:
                break
            
            improved = False
            step = base_step * (0.8 ** iteration)
            
            for i in range(len(current_path)):
                if (time.time() - start_time) > deadline - 30:
                    break
                
                for dim in range(3):
                    for direction in [1, -1]:
                        new_path = current_path.copy()
                        new_path[i, dim] += direction * step
                        
                        # Clip angle to valid range
                        if dim == 2:
                            new_path[i, 2] = np.clip(new_path[i, 2], 5, 85)
                        
                        new_score = evaluate_path_fast(new_path)
                        
                        if new_score > current_score:
                            current_path = new_path
                            current_score = new_score
                            improved = True
        
        return current_path, current_score

    # Run coordinate descent with decreasing base step sizes
    base_steps = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for base_step in base_steps:
        if (time.time() - start_time) > deadline - 30:
            break
        
        path, score = coordinate_descent_optimize(path, max_iterations=150, base_step=base_step)
        
        if score > best_score:
            best_score = score
            best_path = path.copy()

    # Final polish with very small steps
    if (time.time() - start_time) < deadline - 10:
        path, score = coordinate_descent_optimize(path, max_iterations=100, base_step=0.001)
        if score > best_score:
            best_score = score
            best_path = path.copy()

    # Ensure we return the best path found
    if best_score > evaluate_path_fast(path):
        path = best_path.copy()

    return path

# EVOLVE-BLOCK-END
