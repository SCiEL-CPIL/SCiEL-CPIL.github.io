import numpy as np
import time
from scipy.spatial import distance

N = 100

def compute_radius(centers):
    # centers: (N, 3)
    # 1. Min distance to walls
    min_wall_dist = np.min(0.5 - np.abs(centers))
    
    # 2. Min distance between points
    if N > 1:
        min_point_dist = np.min(distance.pdist(centers))
    else:
        min_point_dist = np.inf
        
    return min(min_point_dist, min_wall_dist)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    # Initial grid with jitter
    nx, ny, nz = 4, 5, 5
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    centers = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    # Add small jitter to break symmetry
    centers += np.random.randn(N, 3) * 0.005
    
    # Ensure within bounds
    centers = np.clip(centers, -0.5, 0.5)
    
    best_centers = centers.copy()
    best_r = compute_radius(best_centers)
    
    T = 0.01
    T_min = 1e-8
    alpha = 0.9999 # Slow cooling
    step_size = 0.02
    
    current_r = best_r
    
    # Main SA Loop
    while time.time() < deadline - 5.0:
        T *= alpha
        if T < T_min:
            T = T_min
            
        i = np.random.randint(N)
        old_c = centers[i].copy()
        
        # Move
        centers[i] += np.random.randn(3) * step_size
        centers[i] = np.clip(centers[i], -0.5, 0.5)
        
        new_r = compute_radius(centers)
        
        # Acceptance
        if new_r > current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
        else:
            # Accept with probability exp((new_r - current_r) / T)
            delta = new_r - current_r
            if delta < -50 * T:
                prob = 0.0
            else:
                prob = np.exp(delta / T)
                
            if np.random.rand() < prob:
                current_r = new_r
            else:
                # Reject
                centers[i] = old_c
                
    # Final Polish: Deterministic Local Search
    # Try moving each point in 6 directions
    # Do this for a few iterations or until time runs out
    polish_steps = 0
    # Use a tighter deadline for polish to ensure we return well before 120s
    polish_deadline = deadline - 5.0
    while time.time() < polish_deadline:
        improved = False
        for i in range(N):
            old_c = centers[i].copy()
            for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                # Try small step in this direction
                step = 0.001
                new_c = centers[i].copy()
                new_c[0] += dx * step
                new_c[1] += dy * step
                new_c[2] += dz * step
                
                if np.any(new_c > 0.5) or np.any(new_c < -0.5):
                    continue
                    
                centers[i] = new_c
                r = compute_radius(centers)
                
                if r > best_r:
                    best_r = r
                    best_centers = centers.copy()
                    improved = True
                else:
                    centers[i] = old_c
            
        if not improved:
            break
        polish_steps += 1
        if polish_steps > 50: # Safety break, fewer steps as polish is expensive
            break

    # Final validation
    if not np.all(np.isfinite(best_centers)):
        # Fallback to grid
        centers = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        return centers
        
    return best_centers
