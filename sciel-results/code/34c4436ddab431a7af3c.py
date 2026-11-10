# EVOLVE-BLOCK-START
import numpy as np


def moving_sofa_2d() -> np.ndarray:
    """
    Find a path for the corridor such that the intersection of all placements
    (including the start (0,0,0) and end (-2,0,90)) has maximum area.
    
    The path is a sequence of (x, y, angle) tuples. The evaluator will:
    1. Prepend (0, 0, 0)
    2. Append (-2, 0, 90)
    3. Interpolate between consecutive points with m = max(1, ceil(|a2 - a1|)) steps
    
    We return the intermediate waypoints.
    """
    # The seed path
    path = np.array(
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
    
    # We can try to optimize the path by searching for better waypoints
    # The key insight is that we want to maximize the area of the intersection
    # of all corridor placements.
    
    # Let's try a few different paths and see which gives a better score
    # We'll evaluate them by checking the intersection area
    
    # First, let's implement a function to evaluate a path
    def evaluate_path(path):
        # Add start and end
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        # Interpolate
        interpolated = []
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(m + 1):
                t = k / m
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                interpolated.append([x, y, a])
        
        interpolated = np.array(interpolated)
        
        # Grid points
        xs = np.arange(-3.0, 1.5, 0.01)
        ys = np.arange(-1.5, 1.5, 0.01)
        X, Y = np.meshgrid(xs, ys)
        X = X.flatten()
        Y = Y.flatten()
        
        # Check each placement
        in_all = np.ones(len(X), dtype=bool)
        
        for x, y, a_deg in interpolated:
            a_rad = np.radians(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            # Rotate point by -a
            dx = X - x
            dy = Y - y
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            # Base corridor: union of {0 <= u <= 1, -3 <= v <= 1} and {-3 <= u <= 1, 0 <= v <= 1}
            in_arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            in_corridor = in_arm1 | in_arm2
            
            in_all &= in_corridor
        
        return np.sum(in_all) * 0.0001
    
    # Evaluate the seed path
    seed_score = evaluate_path(path)
    
    # Try some variations
    best_path = path
    best_score = seed_score
    
    # Try moving y up
    for y_shift in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        test_path = path.copy()
        test_path[:, 1] += y_shift
        score = evaluate_path(test_path)
        if score > best_score:
            best_score = score
            best_path = test_path
    
    # Try different x trajectories
    for x_scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
        test_path = path.copy()
        test_path[:, 0] *= x_scale
        score = evaluate_path(test_path)
        if score > best_score:
            best_score = score
            best_path = test_path
    
    # Try adding/removing waypoints
    # Try a simpler path with fewer waypoints
    simple_paths = [
        np.array([[-0.3, 0.2, 30.0], [-0.6, 0.3, 60.0]], dtype=np.float64),
        np.array([[-0.2, 0.1, 20.0], [-0.5, 0.2, 50.0], [-0.8, 0.3, 80.0]], dtype=np.float64),
        np.array([[-0.1, 0.1, 10.0], [-0.3, 0.2, 30.0], [-0.5, 0.3, 50.0], [-0.7, 0.4, 70.0], [-0.9, 0.5, 85.0]], dtype=np.float64),
    ]
    
    for test_path in simple_paths:
        score = evaluate_path(test_path)
        if score > best_score:
            best_score = score
            best_path = test_path
    
    return best_path


# EVOLVE-BLOCK-END
