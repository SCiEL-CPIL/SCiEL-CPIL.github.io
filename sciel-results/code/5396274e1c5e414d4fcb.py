import numpy as np

def construct():
    N = 32
    # Initialize centers on a grid with noise
    side = int(np.ceil(np.sqrt(N)))
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                # Add small noise
                x += np.random.uniform(-0.01, 0.01)
                y += np.random.uniform(-0.01, 0.01)
                centers.append([x, y])
    centers = np.array(centers)
    
    def compute_radii(centers):
        radii = np.zeros(N)
        # Wall distances
        wall_dists = np.min(np.stack([
            centers[:, 0], 
            1 - centers[:, 0], 
            centers[:, 1], 
            1 - centers[:, 1]
        ], axis=1), axis=1)
        
        radii = wall_dists.copy()
        
        for _ in range(30): # Iterations for radius consistency
            # Calculate all pair distances
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            
            # Constraint: r_i <= d_ij - r_j
            limits = dists - radii[np.newaxis, :]
            
            # Exclude self
            np.fill_diagonal(limits, np.inf)
            
            # Min over neighbors
            min_limits = np.min(limits, axis=1)
            
            # Update radii
            new_radii = np.minimum(wall_dists, min_limits)
            new_radii = np.maximum(0, new_radii)
            
            if np.allclose(new_radii, radii, atol=1e-7):
                radii = new_radii
                break
            radii = new_radii
            
        return radii

    radii = compute_radii(centers)
    current_val = np.sum(radii)
    
    # Simulated Annealing
    T = 0.05
    T_min = 0.001
    cooling = 0.999
    steps = 3000
    
    for step in range(steps):
        i = np.random.randint(N)
        # Step size can be adaptive or fixed
        move = np.random.normal(0, 0.05, 2)
        new_centers = centers.copy()
        new_centers[i] += move
        new_centers[i][0] = np.clip(new_centers[i][0], 0.0, 1.0)
        new_centers[i][1] = np.clip(new_centers[i][1], 0.0, 1.0)
        
        new_radii = compute_radii(new_centers)
        new_val = np.sum(new_radii)
        
        delta = new_val - current_val
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            centers = new_centers
            radii = new_radii
            current_val = new_val
            
        T = max(T * cooling, T_min)
        
    # Final Local Search
    # Try small moves
    for _ in range(50):
        improved = False
        for i in range(N):
            for dx, dy in [(0.005, 0), (-0.005, 0), (0, 0.005), (0, -0.005), 
                           (0.0035, 0.0035), (-0.0035, -0.0035), (0.0035, -0.0035), (-0.0035, 0.0035)]:
                new_centers = centers.copy()
                new_centers[i][0] += dx
                new_centers[i][1] += dy
                new_centers[i][0] = np.clip(new_centers[i][0], 0.0, 1.0)
                new_centers[i][1] = np.clip(new_centers[i][1], 0.0, 1.0)
                
                new_radii = compute_radii(new_centers)
                new_val = np.sum(new_radii)
                
                if new_val > current_val:
                    centers = new_centers
                    radii = new_radii
                    current_val = new_val
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [centers[i][0], centers[i][1], radii[i]]
    return result
