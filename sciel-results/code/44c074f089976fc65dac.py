import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # Helper function to compute the radius
    def compute_radius(centers):
        # centers: (N, 3)
        # 1. Pairwise distances
        # Use broadcasting or scipy.spatial.distance
        # For N=50, (50, 50, 3) is small enough for direct calculation
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        # Set diagonal to inf to ignore self-distance
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        # 2. Wall distances
        # Centers must be in [-0.5, 0.5]
        # Distance to wall for coordinate k is 0.5 - abs(center_ik)
        abs_centers = np.abs(centers)
        wall_dists = 0.5 - abs_centers
        min_wall_dist = np.min(wall_dists)
        
        # Radius is half the min pair dist, and also bounded by min wall dist
        # The sphere radius r must satisfy 2r <= min_pair_dist AND r <= min_wall_dist
        # So r = min(min_pair_dist / 2, min_wall_dist)
        return min(min_pair_dist / 2.0, min_wall_dist)

    # Helper function to compute potential energy (to minimize)
    def potential_energy(centers):
        # Add a large penalty if any sphere is outside the cube
        abs_c = np.abs(centers)
        out_of_bounds = np.sum(np.maximum(0, abs_c - 0.5))
        if out_of_bounds > 0:
            return 1e10 + out_of_bounds
            
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1e10) # Avoid division by zero
        
        # Pairwise repulsion
        # We want to maximize min distance, so minimize sum of 1/dist^2
        # Using dist_sq directly for speed
        pair_energy = np.sum(1.0 / dist_sq)
        
        # Wall repulsion
        # Distance to wall w = 0.5 - abs(c)
        # If w is small, energy should be high.
        # Use 1/w^2
        wall_dists = 0.5 - abs_c
        # Clip to avoid division by zero or negative if slightly out (though we check out_of_bounds)
        wall_dists_clipped = np.maximum(wall_dists, 1e-6)
        wall_energy = np.sum(1.0 / (wall_dists_clipped**2))
        
        return pair_energy + wall_energy

    def optimize(centers, max_steps=2000, lr_init=0.01):
        # Simple gradient-free or gradient-based descent
        # We'll use a simple repulsive step method
        # Move each sphere away from others and walls
        
        current_centers = centers.copy()
        best_local_radius = compute_radius(current_centers)
        best_local_centers = current_centers.copy()
        
        lr = lr_init
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            # Calculate forces
            diff = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist_sq, 1e10)
            
            # Force from pairs: F_ij = diff_ij / dist_sq_ij
            # This points from j to i if i is at origin? 
            # diff = c_i - c_j. If c_i > c_j, diff > 0. Force should push i away from j (positive x).
            # So force vector is diff / dist_sq.
            forces = np.sum(diff / (dist_sq[:, :, np.newaxis] + 1e-10), axis=1)
            
            # Force from walls
            # If c_x < -0.5, push right (+x). If c_x > 0.5, push left (-x).
            # Use a smooth potential: F_x = -d/dx (1/(0.5-|x|)^2) ?
            # Simpler: Just use the gradient of the wall energy.
            # Energy = 1/w^2. dE/dx = -2/w^3 * dw/dx.
            # w = 0.5 - |x|. dw/dx = -sign(x).
            # dE/dx = -2/w^3 * (-sign(x)) = 2*sign(x)/w^3.
            # Force = -dE/dx = -2*sign(x)/w^3.
            # If x > 0, sign=1, w=0.5-x. Force = -2/(0.5-x)^3. Pushes left (negative). Correct.
            # If x < 0, sign=-1, w=0.5+x. Force = 2/(0.5+x)^3. Pushes right (positive). Correct.
            
            abs_c = np.abs(current_centers)
            signs = np.sign(current_centers)
            # Handle x=0 carefully, sign is 0, force 0? 
            # If x=0, w=0.5. Force should be 0 due to symmetry? Yes.
            
            w = 0.5 - abs_c
            w_clipped = np.maximum(w, 1e-6)
            
            # Force component for each dimension
            # F = -2 * sign(x) / w^3
            wall_forces = -2.0 * signs / (w_clipped**3)
            
            # Combine forces
            total_forces = forces + wall_forces
            
            # Update positions
            # Scale forces to step size
            # The magnitude of forces can vary, so normalize or use fixed LR
            # Let's use a fixed LR and scale by a factor to keep steps small
            # Heuristic: step = lr * total_forces
            # But forces can be huge if dist is small.
            # Normalize forces?
            force_mags = np.linalg.norm(total_forces, axis=1, keepdims=True)
            # Avoid division by zero
            force_mags = np.maximum(force_mags, 1e-10)
            normalized_forces = total_forces / force_mags
            
            current_centers = current_centers + lr * normalized_forces
            
            # Clamp to cube (soft clamp handled by forces, but hard clamp for safety)
            current_centers = np.clip(current_centers, -0.5, 0.5)
            
            # Check improvement
            current_radius = compute_radius(current_centers)
            if current_radius > best_local_radius:
                best_local_radius = current_radius
                best_local_centers = current_centers.copy()
            
            # Decay learning rate
            lr *= 0.995
            
        return best_local_centers, best_local_radius

    # Initialization Strategies
    
    # Strategy 1: BCC Lattice
    # Create a 3x3x3 grid of points and their BCC centers?
    # Let's just use a 4x4x4 grid and pick 50?
    # Or better: Random
    
    # Let's try a structured grid first to give a good start
    # 3x3x3 = 27 points. Need 50.
    # Let's use a 4x4x4 grid (64 points) and randomly select 50?
    # Or just random.
    
    starts = []
    
    # Start 1: Random
    rng = np.random.default_rng(42)
    starts.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
    
    # Start 2: Grid-based
    # 4x4x4 grid
    ticks = (np.arange(4) + 0.5) / 4 - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Randomly select 50 from 64
    indices = rng.choice(len(grid), size=N, replace=False)
    starts.append(grid[indices])
    
    # Start 3: Another random
    starts.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
    
    # Run optimization on each start
    for i, init in enumerate(starts):
        if time.time() > deadline:
            break
            
        centers, radius = optimize(init, max_steps=3000, lr_init=0.05)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers
            
    # If we have time, do a final polish on the best result
    if time.time() < deadline - 5:
        if best_centers is not None:
            centers, radius = optimize(best_centers, max_steps=5000, lr_init=0.01)
            if radius > best_radius:
                best_radius = radius
                best_centers = centers

    if best_centers is None:
        # Fallback
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
