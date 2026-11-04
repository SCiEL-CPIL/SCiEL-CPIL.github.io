import numpy as np

N = 130

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Generate a high-resolution grid of test points
    M = 50000
    j = np.arange(M) + 0.5
    z_grid = 1.0 - 2.0 * j / M
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * j
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    Y = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    # Helper to compute max angular distance
    def compute_max_dist(P):
        dots = Y @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = np.max(dots, axis=1)
        min_dists = np.arccos(max_dots)
        return np.max(min_dists)
    
    current_dist = compute_max_dist(P)
    
    # Simulated Annealing
    T_start = 0.05
    T_end = 0.0005
    steps = 8000
    
    for it in range(steps):
        T = T_start * (T_end / T_start) ** (it / steps)
        
        # Compute current distances to find worst point
        dots = Y @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = np.max(dots, axis=1)
        min_dists = np.arccos(max_dots)
        
        # Worst point on grid
        worst_idx = np.argmax(min_dists)
        y_worst = Y[worst_idx]
        
        # Nearest point in P
        nearest_p_idx = np.argmax(dots[worst_idx, :])
        
        # Propose move: move nearest_p towards y_worst
        # Use a step size that decays slightly
        alpha = 0.05 + 0.05 * (1.0 - it / steps)
        
        p_old = P[nearest_p_idx].copy()
        v = y_worst - p_old
        p_new = p_old + alpha * v
        norm = np.linalg.norm(p_new)
        if norm < 1e-6:
            continue
        p_new = p_new / norm
        
        # Evaluate new config
        P[nearest_p_idx] = p_new
        new_dist = compute_max_dist(P)
        
        # Accept/Reject
        delta = new_dist - current_dist
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_dist = new_dist
        else:
            P[nearest_p_idx] = p_old
            
    return P
# EVOLVE-BLOCK-END
