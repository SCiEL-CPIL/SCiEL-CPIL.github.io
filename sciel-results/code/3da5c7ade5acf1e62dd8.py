"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def construct():
    # Simulated annealing approach for circle packing
    # Start with a hexagonal lattice initialization
    
    # Generate hexagonal lattice
    rows = 14
    points = []
    s = 1.0
    h = s * np.sqrt(3) / 2
    
    for i in range(rows):
        if i % 2 == 0:
            n_points = 15
            offset = 0.0
        else:
            n_points = 14
            offset = s / 2.0
            
        for j in range(n_points):
            x = j * s + offset
            y = i * h
            points.append([x, y])
            
    points = np.array(points)
    
    # Remove 3 points furthest from center
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    sorted_indices = np.argsort(dists)
    points = points[sorted_indices[:N]]
    
    # Scale and center
    width = points[:, 0].max() - points[:, 0].min()
    height = points[:, 1].max() - points[:, 1].min()
    scale = 0.95 / max(width, height)  # 0.95 to leave margin
    points = (points - center) * scale
    
    # Simulated annealing
    best_pts = points.copy()
    best_r = -np.inf
    
    # Compute initial radius
    def compute_radius(pts):
        # Boundary constraint
        dx = 0.5 - np.abs(pts[:, 0])
        dy = 0.5 - np.abs(pts[:, 1])
        min_boundary = np.min(np.minimum(dx, dy))
        
        # Pairwise distances
        dists = pdist(pts)
        min_pair = np.min(dists) / 2.0
        
        return min(min_boundary, min_pair)
    
    best_r = compute_radius(best_pts)
    current_r = best_r
    
    # Annealing parameters
    T_start = 0.01
    T_end = 0.0001
    n_iter = 5000
    step_size = 0.005
    
    # Time limit
    start_time = time.time()
    time_limit = 100  # seconds
    
    for i in range(n_iter):
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # Linear cooling
        T = T_start * (T_end / T_start) ** (i / n_iter)
        
        # Propose move: pick random point, move it
        idx = np.random.randint(N)
        delta = np.random.normal(0, step_size, 2)
        
        old_pos = best_pts[idx].copy()
        new_pos = old_pos + delta
        
        # Keep within bounds
        new_pos[0] = np.clip(new_pos[0], -0.499, 0.499)
        new_pos[1] = np.clip(new_pos[1], -0.499, 0.499)
        
        best_pts[idx] = new_pos
        new_r = compute_radius(best_pts)
        
        # Accept if better or with probability exp((new_r - current_r) / T)
        if new_r > current_r or np.random.random() < np.exp((new_r - current_r) / T):
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_pts = best_pts.copy()
        else:
            best_pts[idx] = old_pos
    
    return best_pts
# EVOLVE-BLOCK-END
