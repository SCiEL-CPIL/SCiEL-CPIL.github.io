# EVOLVE-BLOCK-START
import numpy as np
import time


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
    (-2, 0, 90), then densifies the path to <=1-degree rotation steps.

    Returns:
        path: np.ndarray of shape (n, 3); each row is (x, y, angle_degrees) for the
            corridor's inner corner.
    """
    start_time = time.time()
    deadline = 620.0  # evaluator kills at 630s, keep safety margin

    # Sofa grid
    x_grid = np.arange(-3, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid)
    P = np.column_stack((X.ravel(), Y.ravel()))  # (N, 2)
    N = P.shape[0]

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

    def check_point_in_corridor_vectorized(P, x, y, a_deg):
        """
        Vectorized check: which points in P are inside the corridor placement at (x, y, a_deg).
        Returns a boolean array of shape (N,).
        """
        a_rad = np.radians(a_deg)
        dx = P[:, 0] - x
        dy = P[:, 1] - y
        cos_a = np.cos(-a_rad)
        sin_a = np.sin(-a_rad)
        u = cos_a * dx - sin_a * dy
        v = sin_a * dx + cos_a * dy
        
        in_r1 = (u >= -1e-9) & (u <= 1 + 1e-9) & (v >= -3 - 1e-9) & (v <= 1 + 1e-9)
        in_r2 = (u >= -3 - 1e-9) & (u <= 1 + 1e-9) & (v >= -1e-9) & (v <= 1 + 1e-9)
        
        return in_r1 | in_r2

    def evaluate_path(path):
        """
        Evaluate the area of the sofa for a given path.
        The sofa is the set of grid points that are inside every placement.
        """
        densified = densify_path(path)
        
        in_sofa = np.ones(N, dtype=bool)
        
        for x, y, a in densified:
            mask = check_point_in_corridor_vectorized(P, x, y, a)
            in_sofa &= mask
            if not in_sofa.any():
                break
        
        return in_sofa.sum() * 0.0001

    def local_search(path, best_score, step_sizes, deadline):
        """
        Perform local search with multiple step sizes.
        Returns (best_path, best_score).
        """
        best_path = path.copy()
        
        for step_size in step_sizes:
            improved = True
            while improved and (time.time() - start_time) < deadline - 10:
                improved = False
                for i in range(len(path)):
                    for _ in range(20):
                        if (time.time() - start_time) > deadline - 10:
                            break
                        
                        new_path = path.copy()
                        new_path[i, 0] += np.random.uniform(-step_size, step_size)
                        new_path[i, 1] += np.random.uniform(-step_size, step_size)
                        new_path[i, 2] += np.random.uniform(-step_size * 2, step_size * 2)
                        new_path[i, 2] = np.clip(new_path[i, 2], 5, 85)
                        
                        score = evaluate_path(new_path)
                        if score > best_score:
                            best_score = score
                            best_path = new_path.copy()
                            path = new_path
                            improved = True
        
        return best_path, best_score

    # Multiple initial paths to try different starting configurations
    initial_paths = [
        np.array([
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ], dtype=np.float64),
        np.array([
            [-0.15, 0.15, 12.0],
            [-0.25, 0.25, 25.0],
            [-0.35, 0.35, 45.0],
            [-0.45, 0.35, 55.0],
            [-0.55, 0.25, 65.0],
            [-0.65, 0.2, 78.0],
            [-0.75, 0.25, 87.0],
        ], dtype=np.float64),
        np.array([
            [-0.05, 0.05, 10.0],
            [-0.15, 0.15, 18.0],
            [-0.25, 0.25, 30.0],
            [-0.35, 0.3, 42.0],
            [-0.45, 0.3, 52.0],
            [-0.55, 0.2, 62.0],
            [-0.65, 0.2, 72.0],
            [-0.75, 0.3, 82.0],
            [-0.8, 0.3, 88.0],
        ], dtype=np.float64),
    ]

    np.random.seed(42)
    best_path = None
    best_score = 0.0

    for init_path in initial_paths:
        if (time.time() - start_time) > deadline - 30:
            break
        
        score = evaluate_path(init_path)
        if score > best_score:
            best_score = score
            best_path = init_path.copy()
        
        # Polish this path
        polished_path, polished_score = local_search(init_path, best_score, 
                                                       [0.1, 0.05, 0.02, 0.01], deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_path = polished_path

    # Final polish on the best path found
    if best_path is not None:
        final_path, final_score = local_search(best_path, best_score,
                                               [0.05, 0.02, 0.01, 0.005], deadline)
        if final_score > best_score:
            best_score = final_score
            best_path = final_path

    # Ensure we return a valid path
    if best_path is None or len(best_path) < 1:
        best_path = initial_paths[0]

    return best_path

# EVOLVE-BLOCK-END
