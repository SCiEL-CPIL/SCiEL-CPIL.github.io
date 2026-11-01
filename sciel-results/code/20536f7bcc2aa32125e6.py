import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initialization: Grid
    # 100 points. 5x5x4 grid.
    # Coordinates in [-0.5, 0.5]
    # Grid ticks for 5 points: -0.4, -0.2, 0, 0.2, 0.4
    # Grid ticks for 4 points: -0.375, -0.125, 0.125, 0.375
    # This creates a 5x5x4 array.
    
    m_x, m_y, m_z = 5, 5, 4
    x_ticks = np.linspace(-0.5 + 1/(2*m_x), 0.5 - 1/(2*m_x), m_x)
    y_ticks = np.linspace(-0.5 + 1/(2*m_y), 0.5 - 1/(2*m_y), m_y)
    z_ticks = np.linspace(-0.5 + 1/(2*m_z), 0.5 - 1/(2*m_z), m_z)
    
    grid = np.array([(x, y, z) for z in z_ticks for y in y_ticks for x in x_ticks])
    points = grid.copy()
    
    # Calculate initial r
    def calc_r(pts):
        # Vectorized min radius calculation
        # Pairwise distances
        # Use broadcasting
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Wall distances
        # dist to wall = 0.5 - max(|x|, |y|, |z|) is WRONG.
        # dist to nearest wall for point (x,y,z) is min(0.5-|x|, 0.5-|y|, 0.5-|z|)
        # = 0.5 - max(|x|, |y|, |z|)
        abs_pts = np.abs(pts)
        max_abs = np.max(abs_pts, axis=1)
        min_wall = np.min(0.5 - max_abs)
        
        return min(min_pair, min_wall)

    current_r = calc_r(points)
    best_r = current_r
    best_points = points.copy()
    
    # SA Parameters
    T_start = 0.02
    T_end = 1e-5
    alpha = 0.999 # Cooling factor
    T = T_start
    
    # Move size scale
    sigma_start = 0.05
    sigma_end = 0.001
    
    step = 0
    while time.time() < deadline:
        # Pick random index
        i = np.random.randint(N)
        
        # Generate move
        # Scale sigma with temperature
        temp_ratio = T / T_start
        sigma = sigma_start * (temp_ratio ** 0.5) # Square root scaling often works well
        
        # Random Gaussian noise
        noise = np.random.normal(0, sigma, size=3)
        
        old_point = points[i].copy()
        points[i] = old_point + noise
        
        # Calculate new r
        new_r = calc_r(points)
        
        # Acceptance
        delta_r = new_r - current_r
        if delta_r >= 0:
            # Improvement
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
        else:
            # Deterioration
            # Accept with probability exp(delta_r / T)
            # Note: delta_r is negative, so exp(negative/positive) < 1
            if np.random.rand() < np.exp(delta_r / T):
                current_r = new_r
            else:
                # Reject
                points[i] = old_point
        
        # Cool down
        T *= alpha
        
        step += 1
        # Optional: Break if T is very low and no recent improvements?
        # For now, just run until deadline.
        
    return best_points
