import numpy as np
import random

def construct():
    N = 50
    # Use a fixed seed for reproducibility in evaluation
    np.random.seed(42)
    
    # 1. Initialization
    # Start with a hexagonal lattice approximation or random.
    # Random is often safer for SA to explore, but hex is a better start.
    # Let's use a hexagonal grid for better initial R.
    
    def hex_init(n):
        # Approximate spacing for n circles in unit square
        # Area per circle ~ 1/n. Hex packing density ~ 0.9069.
        # d ~ sqrt(1/n / 0.9069)
        d = np.sqrt(1.0 / (n * 0.9069))
        
        points = []
        # Generate points in a range larger than the square
        # Spacing in y is d * sqrt(3)/2
        dy = d * np.sqrt(3) / 2
        dx = d
        
        # Determine number of rows/cols needed
        # Rough estimate
        rows = int(np.ceil(1.0 / dy)) + 2
        cols = int(np.ceil(1.0 / dx)) + 2
        
        for r in range(-1, rows):
            for c in range(-1, cols):
                y = r * dy
                x = c * dx
                if r % 2 != 0:
                    x += dx / 2.0
                # Center the grid
                x -= (cols - 1) * dx / 2.0
                y -= (rows - 1) * dy / 2.0
                
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
        
        points = np.array(points)
        
        # If we have more than N, select N randomly
        if len(points) > N:
            indices = np.random.choice(len(points), N, replace=False)
            points = points[indices]
        elif len(points) < N:
            # If fewer, add random points
            while len(points) < N:
                points = np.vstack([points, np.random.uniform(-0.5, 0.5, 2)])
        return points

    centers = hex_init(N)
    
    # 2. Objective Function (Vectorized)
    def get_radius(centers):
        # Centers shape (N, 2)
        # Pairwise distances
        # dist[i, j] = ||c_i - c_j||
        # Use broadcasting
        # c_i: (N, 1, 2), c_j: (1, N, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Exclude diagonal (distance 0)
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        
        # Wall distances
        # dist to wall = 0.5 - |c_ik|
        # For each point, min distance to any of the 4 walls
        # Wall dist for point i = min(0.5 - c_i_x, 0.5 + c_i_x, 0.5 - c_i_y, 0.5 + c_i_y)
        # This is equivalent to 0.5 - max(|c_ix|, |c_iy|) ? No.
        # Distance to left wall: 0.5 - c_ix (if c_ix > -0.5) -> 0.5 - c_ix
        # Distance to right wall: 0.5 + c_ix
        # Min of these two is 0.5 - |c_ix|
        # So min wall dist for point i is 0.5 - max(|c_ix|, |c_iy|)
        
        max_abs_coord = np.max(np.abs(centers), axis=1)
        wall_dists = 0.5 - max_abs_coord
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist, min_wall_dist)

    current_radius = get_radius(centers)
    best_centers = centers.copy()
    best_radius = current_radius
    
    # 3. Simulated Annealing
    # Parameters
    T_start = 0.01
    T_end = 1e-6
    cooling_rate = 0.9995
    max_iter = 50000
    move_scale = 0.05
    
    T = T_start
    
    for i in range(max_iter):
        # Perturb
        # Move a random subset of points
        n_move = np.random.randint(1, 5)
        indices = np.random.choice(N, n_move, replace=False)
        
        new_centers = centers.copy()
        # Add random noise
        noise = np.random.normal(0, move_scale, (n_move, 2))
        new_centers[indices] += noise
        
        # Clip to bounds to ensure validity (though SA can handle slight out-of-bounds with penalty, 
        # clipping is safer for "inside square" constraint if we want to strictly maintain it, 
        # but usually objective function handles it via radius calculation.
        # However, if a point goes outside, wall dist becomes negative. 
        # The problem statement says "inside the square". 
        # If we clip, we might get stuck on boundaries. 
        # Let's allow them to go slightly out, but the radius will drop to negative, 
        # so SA will reject it unless T is high. 
        # Actually, better to clip to [-0.5, 0.5] to keep candidates valid as per contract?
        # "A centre on or outside the square ... make the candidate invalid."
        # So we MUST keep them inside.
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_radius = get_radius(new_centers)
        
        # Acceptance criterion
        if new_radius > current_radius:
            centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
        else:
            # Metropolis criterion
            delta = new_radius - current_radius
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    centers = new_centers
                    current_radius = new_radius
        
        # Cool down
        T *= cooling_rate
        
        # Optional: adapt move_scale
        if i % 1000 == 0:
            move_scale *= 0.95

    return best_centers
