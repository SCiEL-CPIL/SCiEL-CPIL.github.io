import numpy as np
import time

N = 100

def construct():
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 110 # 10s margin
        
        # 1. Initialize: 5x5x4 grid
        # 5*5*4 = 100
        # Spacing in x,y: 1/5 = 0.2 -> r=0.1
        # Spacing in z: 1/4 = 0.25 -> r=0.125
        # But we are in [-0.5, 0.5]
        # x, y ticks: -0.4, -0.2, 0, 0.2, 0.4
        # z ticks: -0.375, -0.125, 0.125, 0.375
        
        x = np.linspace(-0.4, 0.4, 5)
        y = np.linspace(-0.4, 0.4, 5)
        z = np.linspace(-0.375, 0.375, 4)
        
        centers = np.array([(x[i], y[j], z[k]) for i in range(5) for j in range(5) for k in range(4)])
        
        # Calculate initial radius
        def calc_radius(centers):
            # Pairwise distances
            # Use broadcasting
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            
            # Wall distances
            # dist to walls: 0.5 - |c|
            wall_dists = 0.5 - np.abs(centers)
            min_wall_dist = np.min(wall_dists)
            
            return min(min_pair_dist / 2, min_wall_dist)
        
        best_r = calc_radius(centers)
        best_centers = centers.copy()
        
        # SA Parameters
        T = 0.05 # Initial temperature
        T_min = 1e-6
        cooling = 0.999
        step_size = 0.05
        
        current_r = best_r
        current_centers = best_centers.copy()
        
        # Pre-compute indices for efficiency? No, just vectorize.
        
        while time.time() < deadline:
            # Pick a random sphere
            idx = np.random.randint(N)
            
            # Propose move
            move = np.random.normal(0, step_size, 3)
            new_centers = current_centers.copy()
            new_centers[idx] += move
            
            # Check bounds (optional, penalty handles it, but keeping inside helps)
            # If out of bounds, we can just let penalty handle it or clamp.
            # Clamping might get stuck on walls. Let's allow it and let radius drop.
            
            new_r = calc_radius(new_centers)
            
            # Acceptance
            if new_r > current_r:
                current_r = new_r
                current_centers = new_centers
                if current_r > best_r:
                    best_r = current_r
                    best_centers = current_centers.copy()
            else:
                delta = new_r - current_r
                if np.random.random() < np.exp(delta / T):
                    current_r = new_r
                    current_centers = new_centers
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
                # Optional: Restart with higher T if stuck? 
                # Or just keep going.
                
        return best_centers
