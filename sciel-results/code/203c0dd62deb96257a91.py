"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a physics-based relaxation approach (molecular dynamics style):
    1. Initialize points.
    2. Treat circles as particles with repulsive forces if they overlap.
    3. Treat container boundary as a repulsive wall.
    4. Use a symplectic integrator (Velocity Verlet) with damping to relax the system.
    5. Gradually increase the target radius 'r' while relaxing positions.
    6. If relaxation fails (energy doesn't decrease), stop and return the best valid configuration.
    """
    
    def compute_radius(centers):
        """Compute the feasible radius for given centers."""
        dist_to_boundary = 1.0 - np.linalg.norm(centers, axis=1)
        min_boundary = np.min(dist_to_boundary)
        
        if len(centers) < 2:
            min_pair = np.inf
        else:
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
        
        return min(min_boundary, min_pair)

    def calculate_forces(centers, r, k_pair, k_wall, noise_level):
        """Calculate repulsive forces between circles and with the boundary."""
        n = len(centers)
        forces = np.zeros_like(centers)
        
        # Pairwise forces
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Overlap amount: positive if overlapping
        overlap = np.maximum(0, 2 * r - dists)
        
        # Force magnitude is proportional to overlap
        # We need to handle dists=0 carefully, but we set diagonal to inf
        # Force direction is diff / dist
        # To avoid division by zero, we use a safe dist
        safe_dists = np.where(dists < 1e-9, 1e-9, dists)
        
        # Force vector from j to i: k * overlap * (diff / dist)
        # Note: diff[i,j] = p_i - p_j. Force on i should be in direction p_i - p_j if overlapping.
        force_matrix = k_pair * overlap / safe_dists
        force_matrix = np.where(np.eye(n, dtype=bool), 0, force_matrix)
        
        # Sum forces from all j for each i
        pair_forces = np.sum(force_matrix[:, :, np.newaxis] * diff, axis=1)
        forces += pair_forces
        
        # Boundary forces
        dists_to_center = np.linalg.norm(centers, axis=1)
        # If center is too far out (distance > 1 - r), push in
        boundary_overlap = np.maximum(0, dists_to_center - (1.0 - r))
        
        # Direction is towards center: -centers / dists_to_center
        safe_dists_center = np.where(dists_to_center < 1e-9, 1e-9, dists_to_center)
        boundary_forces = -k_wall * boundary_overlap / safe_dists_center[:, np.newaxis] * centers
        
        forces += boundary_forces
        
        # Add random noise for annealing
        if noise_level > 0:
            forces += noise_level * np.random.randn(n, 2)
            
        return forces

    def relax(centers, r, max_iters=5000, k_pair=100.0, k_wall=100.0, dt=0.001, damping=0.9):
        """
        Relax the configuration using Velocity Verlet with damping.
        Returns the relaxed centers and the final energy.
        """
        n = len(centers)
        pos = centers.copy()
        vel = np.zeros_like(pos)
        
        # Initial forces
        f = calculate_forces(pos, r, k_pair, k_wall, 0.0)
        
        min_energy = np.inf
        best_pos = pos.copy()
        
        for i in range(max_iters):
            # Velocity Verlet step
            # v_new = v + 0.5 * dt * (f_old + f_new)
            # p_new = p + 0.5 * dt * (v_old + v_new)
            
            # Simplified symplectic Euler (kick-drift-kick) or just damped Euler
            # Let's use a simple damped Euler for stability and simplicity, 
            # which is often sufficient for these relaxation problems.
            # v = v + dt * f
            # v = v * damping
            # p = p + dt * v
            
            # To make it more "symplectic" and stable, let's use:
            # p += v * dt
            # f_new = forces(p)
            # v += f_new * dt
            # v *= damping
            
            pos += vel * dt
            
            f_new = calculate_forces(pos, r, k_pair, k_wall, 0.0)
            
            vel += f_new * dt
            vel *= damping
            
            # Calculate energy (proxy for quality)
            # Energy = sum of overlaps squared
            dists_to_boundary = 1.0 - np.linalg.norm(pos, axis=1)
            min_boundary = np.min(dists_to_boundary)
            
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
            
            # We want to maximize min(min_boundary, min_pair)
            # So we minimize the violation: max(0, 2r - dists) and max(0, |p| - (1-r))
            
            overlap_pair = np.maximum(0, 2 * r - dists)
            overlap_wall = np.maximum(0, np.linalg.norm(pos, axis=1) - (1.0 - r))
            
            energy = np.sum(overlap_pair ** 2) + np.sum(overlap_wall ** 2)
            
            if energy < min_energy:
                min_energy = energy
                best_pos = pos.copy()
                
            # If energy is very low, we can stop early
            if energy < 1e-6:
                break
                
            # Annealing: add noise that decays
            # We can re-introduce noise in the force calculation or here
            # Let's add noise to velocity
            noise_level = 0.1 * (1.0 - i / max_iters)
            if noise_level > 0:
                vel += noise_level * np.random.randn(n, 2)
                
        return best_pos, min_energy

    def make_initial_hexagonal(n, scale=0.15):
        """Create initial configuration using hexagonal lattice."""
        points = []
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = i * scale + 0.5 * scale * (j % 2)
                y = j * scale * np.sqrt(3) / 2
                if np.sqrt(x * x + y * y) < 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        
        if len(points) > n:
            dists = np.linalg.norm(points, axis=1)
            indices = np.argsort(dists)[:n]
            points = points[indices]
        elif len(points) < n:
            while len(points) < n:
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0, 0.9)
                points = np.vstack([points, [radius * np.cos(angle), radius * np.sin(angle)]])
        
        return points

    # Main logic
    np.random.seed(42)
    
    # Start with a small radius and increase
    r_current = 0.05
    r_step = 0.005
    max_r = 0.15
    
    # Initialize with a reasonable layout
    centers = make_initial_hexagonal(N, scale=0.1)
    
    best_centers = centers
    best_r = compute_radius(centers)
    
    # Relaxation loop
    while r_current < max_r:
        # Relax the configuration for the current radius
        centers_relaxed, energy = relax(centers, r_current, max_iters=2000)
        
        # Check if the relaxed configuration is valid for r_current
        # i.e., no overlaps and inside boundary
        actual_r = compute_radius(centers_relaxed)
        
        if actual_r >= r_current - 1e-6:
            # Success, update best and increase r
            best_centers = centers_relaxed
            best_r = actual_r
            centers = centers_relaxed
            r_current += r_step
        else:
            # Failure, try a smaller step or break
            # Try a smaller step
            r_step /= 2
            if r_step < 0.0001:
                break
            r_current += r_step
            
    # Final relaxation with the best r found to polish
    if best_r > 0:
        # Try to optimize slightly above best_r? No, just polish at best_r
        centers_polished, _ = relax(best_centers, best_r, max_iters=5000, k_pair=1000.0, k_wall=1000.0)
        final_r = compute_radius(centers_polished)
        if final_r > best_r:
            best_centers = centers_polished
            best_r = final_r
            
    return best_centers
# EVOLVE-BLOCK-END
