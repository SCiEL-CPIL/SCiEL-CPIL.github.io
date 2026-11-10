# EVOLVE-BLOCK-START
import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    """
    Find a path for the moving sofa problem that maximizes the sofa area.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill
    
    # Grid setup
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    n_points = grid_points.shape[0]
    
    # Precompute the base corridor check function
    # Base: {0<=u<=1, -3<=v<=1} OR {-3<=u<=1, 0<=v<=1}
    
    def check_in_base(u, v):
        """Check if (u,v) is in the base L-shape corridor."""
        in_v = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        in_h = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        return in_v | in_h
    
    def evaluate_path(path):
        """
        Evaluate the sofa area for a given path.
        path: array of shape (n, 3) with (x, y, angle_degrees)
        Returns the raw value (number of sofa points * 0.0001)
        """
        # Build the full path with endpoints and interpolation
        full_path = [np.array([0.0, 0.0, 0.0])]
        
        for i in range(len(path)):
            full_path.append(path[i])
        
        full_path.append(np.array([-2.0, 0.0, 90.0]))
        
        # Convert to array
        full_path = np.array(full_path)
        
        # Interpolate between consecutive points
        interp_paths = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            if m > 1:
                ts = np.linspace(0, 1, m + 1)[1:]  # Skip t=0 (already added)
                for t in ts:
                    interp = p1 + t * (p2 - p1)
                    interp_paths.append(interp)
        
        interp_paths = np.array(interp_paths)
        
        # For each grid point, check if it's in ALL corridor placements
        # Start with all True
        inside_all = np.ones(n_points, dtype=bool)
        
        for placement in interp_paths:
            x, y, a_deg = placement
            a_rad = np.radians(a_deg)
            
            # R(-a) rotation matrix
            c = np.cos(-a_rad)
            s = np.sin(-a_rad)
            
            # w - (x, y)
            dx = grid_points[:, 0] - x
            dy = grid_points[:, 1] - y
            
            # Rotate
            u = c * dx - s * dy
            v = s * dx + c * dy
            
            # Check if in base
            in_v = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_h = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            in_base = in_v | in_h
            
            inside_all &= in_base
        
        n_inside = np.sum(inside_all)
        return n_inside * 0.0001
    
    # Initial path from the seed
    initial_path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ])
    
    best_path = initial_path.copy()
    best_score = evaluate_path(best_path)
    
    # Try to improve with random perturbations
    np.random.seed(42)
    
    n_iter = 0
    while time.time() < deadline:
        # Perturb the path
        perturbed = best_path.copy()
        for i in range(len(perturbed)):
            perturbed[i, 0] += np.random.normal(0, 0.1)  # x
            perturbed[i, 1] += np.random.normal(0, 0.1)  # y
            perturbed[i, 2] += np.random.normal(0, 3.0)   # angle
        
        # Ensure angles are in a reasonable range
        perturbed[:, 2] = np.clip(perturbed[:, 2], 5, 85)
        
        score = evaluate_path(perturbed)
        
        if score > best_score:
            best_score = score
            best_path = perturbed.copy()
        
        n_iter += 1
    
    return best_path

# EVOLVE-BLOCK-END
