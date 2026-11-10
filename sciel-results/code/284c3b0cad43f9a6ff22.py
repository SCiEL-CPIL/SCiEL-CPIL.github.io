import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    # Grid definition
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    x_grid, y_grid = np.meshgrid(xs, ys)
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    n_points = len(x_flat)
    
    # Time management
    start_time = time.time()
    time_limit = 580  # Safety margin before 630s kill
    
    def evaluate_path(path):
        """path: list of (x, y, a). Returns area score."""
        full_path = [(0.0, 0.0, 0.0)]
        for p in path:
            full_path.append(p)
        full_path.append((-2.0, 0.0, 90.0))
        
        # Interpolate
        interp_points = []
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(m):
                t = (k + 1) / m
                x = p1[0] + (p2[0] - p1[0]) * t
                y = p1[1] + (p2[1] - p1[1]) * t
                a = p1[2] + (p2[2] - p1[2]) * t
                interp_points.append((x, y, a))
        interp_points.append(full_path[-1])
        
        mask = np.ones(n_points, dtype=bool)
        
        for (cx, cy, ca_deg) in interp_points:
            ca = np.deg2rad(ca_deg)
            cos_a = np.cos(ca)
            sin_a = np.sin(ca)
            
            dx = x_flat - cx
            dy = y_flat - cy
            
            # R(-a) rotation
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a
            
            in_arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            
            in_corridor = in_arm1 | in_arm2
            mask &= in_corridor
            
            if not np.any(mask):
                return 0.0
        
        return np.sum(mask) * 0.0001

    # Initial path: use inspiration 1's hand-tuned path as starting point
    # This has been shown to give good baseline
    initial_path = [
        (-0.1, 0.1, 15.0),
        (-0.2, 0.2, 20.0),
        (-0.3, 0.3, 34.0),
        (-0.3, 0.3, 40.0),
        (-0.4, 0.3, 50.0),
        (-0.5, 0.2, 55.0),
        (-0.6, 0.2, 75.0),
        (-0.7, 0.3, 86.0),
    ]
    
    best_score = evaluate_path(initial_path)
    best_path = initial_path[:]
    current_path = initial_path[:]
    
    # Local search with adaptive step sizes
    # Perturb x, y, and angle (with small angle perturbations)
    np.random.seed(42)
    epsilon_xy = 0.15
    epsilon_a = 3.0
    no_improve = 0
    max_no_improve = 30
    
    while time.time() - start_time < time_limit:
        # Pick random control point
        idx = np.random.randint(len(current_path))
        p = current_path[idx]
        
        # Perturb all three coordinates
        new_x = p[0] + np.random.normal(0, epsilon_xy)
        new_y = p[1] + np.random.normal(0, epsilon_xy)
        new_a = p[2] + np.random.normal(0, epsilon_a)
        
        # Constrain angle to [0, 90]
        new_a = np.clip(new_a, 0.0, 90.0)
        
        new_p = (new_x, new_y, new_a)
        test_path = current_path[:]
        test_path[idx] = new_p
        
        score = evaluate_path(test_path)
        
        if score > best_score:
            best_score = score
            best_path = test_path
            current_path = test_path
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                # Shrink step sizes
                epsilon_xy *= 0.85
                epsilon_a *= 0.85
                no_improve = 0
                # If step sizes get too small, try a restart from best
                if epsilon_xy < 0.01 and epsilon_a < 0.5:
                    current_path = best_path[:]
                    epsilon_xy = 0.3
                    epsilon_a = 5.0
    
    return np.array(best_path)
