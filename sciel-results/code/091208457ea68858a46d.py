import numpy as np

N = 50

def construct():
    """
    Finds the optimal packing of N=50 circles in a unit circle using Simulated Annealing.
    Returns an array of shape (N, 2) with the centers.
    """
    
    def compute_radius(centers):
        """Compute the maximum radius for the given centers."""
        # Distance from origin
        dist_origin = np.linalg.norm(centers, axis=1)
        boundary_constraint = 1.0 - np.max(dist_origin)
        
        # Pairwise distances
        # Efficient calculation using broadcasting
        # diff shape (N, N, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        # Set diagonal to infinity to ignore self-distance
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_dist = np.sqrt(np.min(dist_sq))
        
        pair_constraint = min_pair_dist / 2.0
        
        return min(boundary_constraint, pair_constraint)

    def anneal(initial_centers, t_start=1.0, t_end=0.001, iterations=50000, cooling_rate=0.9995):
        current_centers = initial_centers.copy()
        current_r = compute_radius(current_centers)
        best_centers = current_centers.copy()
        best_r = current_r
        
        t = t_start
        for i in range(iterations):
            # Pick a random point to move
            idx = np.random.randint(N)
            
            # Generate a random move
            # Scale move size with temperature
            move_size = t * 0.1
            delta = np.random.normal(0, move_size, size=2)
            
            new_centers = current_centers.copy()
            new_centers[idx] += delta
            
            # Check if new position is within reasonable bounds (e.g., inside radius 1.5)
            # to avoid points flying off to infinity
            if np.linalg.norm(new_centers[idx]) > 1.5:
                continue
                
            new_r = compute_radius(new_centers)
            
            # Acceptance criterion
            if new_r > current_r or np.random.rand() < np.exp((new_r - current_r) / t):
                current_centers = new_centers
                current_r = new_r
                
                if current_r > best_r:
                    best_r = current_r
                    best_centers = current_centers.copy()
            
            t *= cooling_rate
            
        return best_centers, best_r

    # Multiple restarts to escape local minima
    best_overall_r = 0.0
    best_overall_centers = None
    
    # Use a few different initialization strategies
    num_restarts = 5
    
    for restart in range(num_restarts):
        if restart == 0:
            # Hexagonal lattice initialization
            # Approximate spacing
            spacing = 0.2 
            points = []
            # Generate a grid of hex points
            for i in range(-5, 6):
                for j in range(-5, 6):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3) / 2
                    if np.sqrt(x**2 + y**2) < 0.9:
                        points.append([x, y])
            points = np.array(points)
            # If we have more than N, randomly select N
            if len(points) > N:
                idxs = np.random.choice(len(points), N, replace=False)
                initial = points[idxs]
            else:
                # Pad with random points if needed
                while len(points) < N:
                    r = np.random.rand() * 0.8
                    theta = np.random.rand() * 2 * np.pi
                    points.append([r * np.cos(theta), r * np.sin(theta)])
                initial = np.array(points[:N])
        else:
            # Random initialization
            initial = np.random.uniform(-0.8, 0.8, size=(N, 2))
            # Ensure they are inside
            norms = np.linalg.norm(initial, axis=1)
            initial[norms > 0.8] *= 0.8 / norms[norms > 0.8]

        centers, r = anneal(initial, t_start=0.5, t_end=0.0001, iterations=20000, cooling_rate=0.999)
        
        if r > best_overall_r:
            best_overall_r = r
            best_overall_centers = centers

    # Final polish: A short run of gradient-like steps or just return best found
    # The SA should be sufficient.
    
    return best_overall_centers

