import numpy as np
import time

N = 130

def construct():
    start_time = time.time()
    time_limit = 120.0  # Leave some margin before 150s
    
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, np.newaxis]
    
    # 2. Generate test points for covering radius estimation
    n_test = 20000
    rng = np.random.default_rng(42)
    u = rng.random((n_test, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    
    best_pts = pts.copy()
    best_score = -np.inf  # We want to maximize the proxy score
    
    step_size = 0.1
    
    while time.time() - start_time < time_limit:
        dots = np.dot(u, pts.T)  # shape (n_test, N)
        max_dots = np.max(dots, axis=1)  # shape (n_test,)
        
        # The test point that determines the covering radius is the one with the smallest max_dot
        worst_idx = np.argmin(max_dots)
        y = u[worst_idx]
        
        # Find the config point closest to y (max dot product with y)
        closest_idx = np.argmax(dots[worst_idx, :])
        
        p = pts[closest_idx]
        direction = y - p
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 1e-10:
            new_p = p + step_size * direction / dir_norm
            new_p_norm = np.linalg.norm(new_p)
            if new_p_norm > 1e-10:
                pts[closest_idx] = new_p / new_p_norm
        
        # Current proxy score: min_i (max_j dots[i, j])
        # We want to maximize this value
        current_score_proxy = np.min(max_dots)
        
        if current_score_proxy > best_score:
            best_score = current_score_proxy
            best_pts = pts.copy()
            
        # Decay step size
        step_size *= 0.99
        if step_size < 0.001:
            step_size = 0.001
            
    return best_pts
