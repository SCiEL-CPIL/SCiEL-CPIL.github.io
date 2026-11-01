import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Initialization: 5x5x4 grid
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.375, 0.375, 4)
    centers = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    # Precompute indices for vectorized force?
    # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # This allocates (100, 100, 3) array. 30k floats. 240KB. Fine.
    
    step_size = 0.05
    decay = 0.999
    
    for i in range(20000):
        if time.time() > deadline:
            break
            
        # Compute forces
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq[dist_sq < 1e-12] = 1e-12
        
        # Force = diff / dist_sq
        # Sum over j
        forces = np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
        
        # Wall forces
        wall_dist = 0.5 - np.abs(centers)
        wall_dist[wall_dist < 1e-6] = 1e-6
        wall_forces = -np.sign(centers) / (wall_dist**2)[:, :, np.newaxis]
        
        total_forces = forces + wall_forces
        
        # Normalize and update
        norms = np.linalg.norm(total_forces, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1e-6
        centers += step_size * total_forces / norms
        centers = np.clip(centers, -0.4999, 0.4999)
        
        step_size *= decay

    return centers
