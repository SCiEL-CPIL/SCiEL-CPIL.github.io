import numpy as np
import time

def construct():
    N = 50
    start_time = time.time()
    time_limit = 150.0
    safety_margin = 5.0  # Leave 5 seconds for final checks and return

    # 1. Initialization: Hexagonal Lattice
    # Generate a hexagonal lattice of points inside the unit circle
    # Adjust spacing to fit N points
    
    def generate_hex_lattice(N):
        # Estimate spacing
        # Area of circle = pi. Area of hex cell ~ sqrt(3)/2 * s^2.
        # N * sqrt(3)/2 * s^2 ~ pi => s ~ sqrt(2*pi / (N*sqrt(3)))
        s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
        
        points = []
        # Generate points in a grid large enough to cover unit circle
        # Hex grid:
        # x = i * s + (j % 2) * s / 2
        # y = j * s * sqrt(3) / 2
        
        max_range = int(1.0 / s) + 2
        for j in range(-max_range, max_range + 1):
            for i in range(-max_range, max_range + 1):
                x = i * s + (j % 2) * s * 0.5
                y = j * s * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    points.append((x, y))
        
        points = np.array(points)
        
        # Sort by distance from center to pick the innermost N points
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists, kind='stable')
        points = points[indices[:N]]
        
        return points

    centers = generate_hex_lattice(N)
    
    # Calculate initial radius
    def calc_radius(centers):
        N = len(centers)
        min_dist = np.inf
        # Pairwise distances
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_dist:
                    min_dist = d
        min_dist /= 2.0
        
        # Boundary distance
        max_norm = np.max(np.linalg.norm(centers, axis=1))
        bound_dist = 1.0 - max_norm
        
        return min(min_dist, bound_dist)

    best_r = calc_radius(centers)
    best_centers = centers.copy()
    
    # 2. Optimization Loop
    delta_r = 1e-4  # Initial step size for radius increase
    min_delta_r = 1e-8
    step_size = 0.05  # Relaxation step size
    max_relax_steps = 100
    jitter_std = 0.01
    
    rng = np.random.default_rng(42)
    
    def relax(centers, r_target, step_size, max_steps):
        """
        Attempts to adjust centers to satisfy constraints for radius r_target.
        Returns True if successful, False otherwise.
        Modifies centers in place.
        """
        N = len(centers)
        
        for _ in range(max_steps):
            # Calculate forces
            forces = np.zeros_like(centers)
            
            # Pairwise repulsion
            # Vectorized calculation
            # diff[i,j] = centers[i] - centers[j]
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            # Avoid division by zero
            np.fill_diagonal(dists, 1.0)
            
            # Overlap amount
            # We want d >= 2*r_target
            overlap = np.maximum(0.0, 2.0 * r_target - dists)
            
            # Force direction: (centers[i] - centers[j]) / dists[i,j]
            # Force magnitude: overlap
            # F_i = sum_j overlap[i,j] * (c_i - c_j) / d_ij
            # Note: diff[i,j] is c_i - c_j.
            # So force vector is diff * (overlap / dists)
            
            force_weights = overlap / dists
            np.fill_diagonal(force_weights, 0.0)
            
            # Sum over j for each i
            forces += np.sum(force_weights[:, :, np.newaxis] * diff, axis=1)
            
            # Boundary repulsion
            # We want |c_i| <= 1 - r_target
            norms = np.linalg.norm(centers, axis=1)
            boundary_overlap = np.maximum(0.0, norms - (1.0 - r_target))
            
            # Force direction: -centers / norms
            # Force magnitude: boundary_overlap
            # Avoid div by 0
            safe_norms = np.where(norms > 1e-8, norms, 1.0)
            boundary_forces = -centers / safe_norms[:, np.newaxis] * boundary_overlap[:, np.newaxis]
            
            forces += boundary_forces
            
            # Update positions
            centers += step_size * forces
            
            # Check convergence (optional, but good for early exit)
            if np.max(np.linalg.norm(forces, axis=1)) < 1e-6:
                break
        
        # Final validation
        return is_valid(centers, r_target)

    def is_valid(centers, r):
        N = len(centers)
        # Check finite values
        if not np.all(np.isfinite(centers)):
            return False
        
        # Check boundary
        norms = np.linalg.norm(centers, axis=1)
        if np.any(norms > 1.0 - r + 1e-9):
            return False
            
        # Check pairwise distances
        # Vectorized
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_dist = np.min(dists)
        if min_dist < 2.0 * r - 1e-9:
            return False
            
        return True

    while time.time() - start_time < time_limit - safety_margin:
        r_target = best_r + delta_r
        
        # Jitter to escape local minima
        # Only jitter if we've had some successes or to explore
        # Small jitter
        centers_trial = best_centers + rng.normal(0, jitter_std, size=(N, 2))
        
        # Relax
        success = relax(centers_trial, r_target, step_size, max_relax_steps)
        
        if success:
            # Verify strictly
            if is_valid(centers_trial, r_target):
                best_r = r_target
                best_centers = centers_trial.copy()
                # Slightly increase delta_r if we are progressing well?
                # Or keep it constant.
                # Let's keep it constant for stability, maybe reduce jitter slightly
                jitter_std *= 0.99
            else:
                # Relaxation claimed success but validation failed
                delta_r *= 0.9
        else:
            # Relaxation failed
            delta_r *= 0.9
            if delta_r < min_delta_r:
                break

    return best_centers
