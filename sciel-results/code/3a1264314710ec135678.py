import numpy as np

N = 50

def construct():
    np.random.seed(42)
    
    # Initialize with a jittered grid in a smaller box to ensure wall distance is not the bottleneck initially
    # Grid in [-0.3, 0.3]
    n_grid = 4
    coords = np.linspace(-0.3, 0.3, n_grid)
    grid_points = np.array([[x, y, z] for x in coords for y in coords for z in coords])
    
    # Select 50 points
    indices = np.random.choice(len(grid_points), N, replace=False)
    centers = grid_points[indices].copy()
    
    # Add small jitter
    centers += np.random.normal(0, 0.01, (N, 3))
    
    # Clip to bounds (should not be needed if jitter is small, but safe)
    centers = np.clip(centers, -0.49, 0.49)
    
    def get_radius(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        min_wall_dist = np.min([
            np.min(centers[:, k] + 0.5) for k in range(3)
        ])
        min_wall_dist = min(min_wall_dist, np.min([
            np.min(0.5 - centers[:, k]) for k in range(3)
        ]))
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    current_radius = get_radius(centers)
    best_radius = current_radius
    best_centers = centers.copy()
    
    # Simulated Annealing
    T = 0.05  # Start with larger temp to explore
    T_min = 1e-8
    alpha = 0.9995
    iterations = 100000
    
    current_centers = centers.copy()
    
    for i in range(iterations):
        idx = np.random.randint(N)
        noise = np.random.normal(0, T, 3)
        new_centers = current_centers.copy()
        new_centers[idx] += noise
        new_centers[idx] = np.clip(new_centers[idx], -0.49, 0.49)
        
        new_radius = get_radius(new_centers)
        delta_radius = new_radius - current_radius
        
        if delta_radius > 0 or np.random.rand() < np.exp(delta_radius / T):
            current_centers = new_centers
            current_radius = new_radius
            
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = current_centers.copy()
        
        T *= alpha
        if T < T_min:
            break
            
    # Final Polish
    T = 1e-4
    for i in range(50000):
        idx = np.random.randint(N)
        noise = np.random.normal(0, T, 3)
        new_centers = best_centers.copy()
        new_centers[idx] += noise
        new_centers[idx] = np.clip(new_centers[idx], -0.49, 0.49)
        
        new_radius = get_radius(new_centers)
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = new_centers.copy()
        T *= 0.9999

    return best_centers
