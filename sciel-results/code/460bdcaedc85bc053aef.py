import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    best_pos = None
    best_score = 0.0
    
    def get_score(pos):
        # pos is (N, 3)
        # Calculate min pairwise distance
        dists = []
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                dists.append(d)
        min_pair = min(dists) if dists else 0.0
        
        # Calculate min wall distance
        # Cube is [-0.5, 0.5]
        # Dist to wall x: 0.5 - abs(x)
        wall_dists = []
        for i in range(N):
            for k in range(3):
                d_wall = 0.5 - abs(pos[i][k])
                wall_dists.append(d_wall)
        min_wall = min(wall_dists) if wall_dists else 0.0
        
        return min(min_pair, min_wall) / 2.0

    def energy(pos_flat):
        pos = pos_flat.reshape(N, 3)
        # Pairwise repulsion
        e = 0.0
        # Use vectorized distance calculation for speed
        # diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=2)
        # np.fill_diagonal(dists, np.inf)
        # e += np.sum(1.0 / (dists**2))
        
        # Loop is slower but clearer. Let's use vectorized.
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        # Clip to avoid division by zero
        dists = np.maximum(dists, 1e-6)
        e += np.sum(1.0 / (dists**4))
        
        # Wall repulsion
        # Distance to each wall face
        # x: 0.5 - x, 0.5 + x
        # y: 0.5 - y, 0.5 + y
        # z: 0.5 - z, 0.5 + z
        walls = np.abs(pos)
        # dist to wall is 0.5 - wall_coord
        d_walls = 0.5 - walls
        d_walls = np.maximum(d_walls, 1e-6)
        e += np.sum(1.0 / (d_walls**4))
        
        return e

    # Initial random restarts
    n_restarts = 0
    max_restarts = 50
    
    # We will alternate between random starts and perturbed best starts
    for i in range(max_restarts):
        if time.time() > deadline:
            break
            
        if i == 0:
            # Start with a grid-like structure for the first run?
            # Or just random.
            pos = np.random.uniform(-0.45, 0.45, (N, 3))
        elif best_pos is not None and i % 2 == 0:
            # Perturb best
            noise = np.random.normal(0, 0.05, (N, 3))
            pos = best_pos + noise
            # Project back into cube
            pos = np.clip(pos, -0.49, 0.49)
        else:
            pos = np.random.uniform(-0.45, 0.45, (N, 3))
            
        pos_flat = pos.flatten()
        
        # Optimize
        try:
            res = minimize(energy, pos_flat, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
            opt_pos = res.x.reshape(N, 3)
            
            # Ensure points are strictly inside (clip with small margin)
            opt_pos = np.clip(opt_pos, -0.4999, 0.4999)
            
            score = get_score(opt_pos)
            
            if score > best_score:
                best_score = score
                best_pos = opt_pos
                
        except Exception as e:
            pass

    if best_pos is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]

    return best_pos
