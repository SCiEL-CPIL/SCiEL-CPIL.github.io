"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin before 120s limit
    
    def compute_score(points):
        """Compute the minimum radius for given points."""
        if len(points) != N:
            return 0.0
        # Check all points are finite
        if not np.all(np.isfinite(points)):
            return 0.0
        # Check all points are inside unit circle (strictly)
        dists = np.sqrt(np.sum(points**2, axis=1))
        if np.any(dists >= 1.0):
            return 0.0
        # Compute all pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=2))
        # Minimum pairwise distance
        np.fill_diagonal(dist_mat, np.inf)
        min_pair_dist = np.min(dist_mat)
        # Minimum distance to boundary
        min_boundary = np.min(1.0 - dists)
        return min(min_pair_dist / 2.0, min_boundary)
    
    def hex_lattice_init():
        """Generate initial hexagonal lattice packing."""
        # Estimate spacing for N circles
        # Area of unit circle = pi, area per circle ~ pi/N
        # Spacing ~ sqrt(pi/N) * factor
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        
        # Generate hex lattice points
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= (1.0 - spacing/2.0)**2:
                    rows.append([x, y])
        
        points = np.array(rows)
        if len(points) < N:
            # Adjust spacing and retry
            spacing *= 0.95
            k = int(1.0 / spacing) + 3
            rows = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    if x*x + y*y <= (1.0 - spacing/2.0)**2:
                        rows.append([x, y])
            points = np.array(rows)
        
        if len(points) >= N:
            # Select N points closest to center
            dists = np.sqrt(np.sum(points**2, axis=1))
            idx = np.argsort(dists, kind="stable")[:N]
            return points[idx]
        return points
    
    def optimize(points, max_iters=5000):
        """Optimize packing using gradient-free approach."""
        best_score = compute_score(points)
        best_points = points.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random perturbation
            step_size = 0.01 * (1.0 - it / max_iters) + 0.001
            perturbed = points.copy()
            idx = np.random.randint(0, N)
            perturbed[idx] += np.random.randn(2) * step_size
            
            new_score = compute_score(perturbed)
            if new_score > best_score:
                best_score = new_score
                best_points = perturbed.copy()
        
        return best_points, best_score
    
    # Initialize with hex lattice
    points = hex_lattice_init()
    best_points = points.copy()
    best_score = compute_score(points)
    
    # Optimize
    points, score = optimize(points, max_iters=3000)
    if score > best_score:
        best_score = score
        best_points = points
    
    # Second round of optimization with smaller steps
    points, score = optimize(best_points, max_iters=2000)
    if score > best_score:
        best_score = score
        best_points = points
    
    return best_points
# EVOLVE-BLOCK-END
