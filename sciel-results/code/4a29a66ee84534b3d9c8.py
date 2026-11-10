# EVOLVE-BLOCK-START
import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    """
    Moves a sofa through an L-shaped corridor.
    Strategy: Dense waypoint optimization with vectorized evaluation.
    We represent the path by 89 waypoints (angles 1 to 89 degrees).
    We use a vectorized evaluation function to check all waypoints simultaneously.
    We perform coordinate descent + adaptive random search on the (x, y) coordinates.
    """
    
    # 1. Setup Grid for Evaluation
    x_grid = np.arange(-3, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid)
    P = np.column_stack((X.ravel(), Y.ravel()))
    N_points = P.shape[0]
    wx = P[:, 0]
    wy = P[:, 1]
    
    start_state = np.array([0.0, 0.0, 0.0])
    end_state = np.array([-2.0, 0.0, 90.0])
    
    def evaluate_path(path_array):
        """
        Evaluates the area of the sofa for a given path.
        path_array: np.ndarray of shape (n, 3)
        Returns: float area
        """
        if path_array.size == 0:
            full_path = [start_state, end_state]
        else:
            full_path = [start_state] + path_array.tolist() + [end_state]
        
        mask = np.ones(N_points, dtype=bool)
        
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i + 1]
            
            a1 = p1[2]
            a2 = p2[2]
            
            da = a2 - a1
            steps = max(1, int(np.ceil(abs(da))))
            
            if steps == 1:
                eval_points = [p2]
            else:
                eval_points = []
                for j in range(1, steps + 1):
                    t = j / steps
                    nx = p1[0] + t * (p2[0] - p1[0])
                    ny = p1[1] + t * (p2[1] - p1[1])
                    na = a1 + t * da
                    eval_points.append([nx, ny, na])
            
            for p in eval_points:
                x, y, a_deg = p
                a_rad = np.deg2rad(a_deg)
                
                cos_a = np.cos(a_rad)
                sin_a = np.sin(a_rad)
                
                dx = wx - x
                dy = wy - y
                
                u = dx * cos_a + dy * sin_a
                v = -dx * sin_a + dy * cos_a
                
                in_r1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
                in_r2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
                
                in_corridor = in_r1 | in_r2
                
                mask &= in_corridor
                
                if not np.any(mask):
                    return 0.0
                    
        return float(np.sum(mask)) * 0.0001

    def generate_path_from_waypoints(waypoints):
        """
        waypoints: np.ndarray of shape (89, 2)
        Returns: np.ndarray of shape (89, 3)
        """
        angles = np.arange(1, 90, dtype=float)
        path = np.column_stack([waypoints[:, 0], waypoints[:, 1], angles])
        return path

    def initial_waypoints():
        """
        Generate initial waypoints using a sinusoidal base path.
        """
        angles = np.arange(1, 90, dtype=float)
        t = angles / 90.0
        # Base: linear from (0,0) to (-2,0) with sinusoidal bulge
        x = -2.0 * t + 0.8 * np.sin(np.pi * t)
        y = 1.2 * np.sin(np.pi * t)
        return np.column_stack([x, y])

    # 2. Search
    start_time = time.time()
    deadline = start_time + 590.0
    
    np.random.seed(42)
    
    # Initialize with a good base
    best_waypoints = initial_waypoints()
    best_score = evaluate_path(generate_path_from_waypoints(best_waypoints))
    
    current_waypoints = best_waypoints.copy()
    current_score = best_score
    
    # Phase 1: Coordinate Descent
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    
    for step in step_sizes:
        improved = False
        for _ in range(30):  # 30 iterations per step size
            if time.time() > deadline:
                break
            
            # Try perturbing a random waypoint
            idx = np.random.randint(0, 89)
            coord = np.random.randint(0, 2)  # 0 for x, 1 for y
            
            new_waypoints = current_waypoints.copy()
            new_waypoints[idx, coord] += np.random.uniform(-step, step)
            
            path = generate_path_from_waypoints(new_waypoints)
            score = evaluate_path(path)
            
            if score > current_score:
                current_score = score
                current_waypoints = new_waypoints
                improved = True
        
        if not improved:
            break
        
        if current_score > best_score:
            best_score = current_score
            best_waypoints = current_waypoints.copy()
    
    # Phase 2: Adaptive Random Search
    # Use larger steps to escape local optima, then refine
    for _ in range(50):
        if time.time() > deadline:
            break
        
        # Random restarts from best
        if np.random.random() < 0.1:
            new_waypoints = best_waypoints.copy()
        else:
            new_waypoints = current_waypoints.copy()
        
        # Perturb multiple waypoints
        n_perturb = np.random.randint(1, 5)
        for _ in range(n_perturb):
            idx = np.random.randint(0, 89)
            coord = np.random.randint(0, 2)
            step = np.random.uniform(0.01, 0.1)
            new_waypoints[idx, coord] += np.random.uniform(-step, step)
        
        path = generate_path_from_waypoints(new_waypoints)
        score = evaluate_path(path)
        
        if score > current_score:
            current_score = score
            current_waypoints = new_waypoints
            if score > best_score:
                best_score = score
                best_waypoints = new_waypoints.copy()
    
    # Phase 3: Final Polish with very small steps
    if time.time() < deadline - 10:
        for _ in range(20):
            if time.time() > deadline:
                break
            
            idx = np.random.randint(0, 89)
            coord = np.random.randint(0, 2)
            step = 0.001
            
            new_waypoints = current_waypoints.copy()
            new_waypoints[idx, coord] += np.random.uniform(-step, step)
            
            path = generate_path_from_waypoints(new_waypoints)
            score = evaluate_path(path)
            
            if score > current_score:
                current_score = score
                current_waypoints = new_waypoints
                if score > best_score:
                    best_score = score
                    best_waypoints = new_waypoints.copy()
    
    # Generate final path
    final_path = generate_path_from_waypoints(best_waypoints)
    return final_path

# EVOLVE-BLOCK-END
