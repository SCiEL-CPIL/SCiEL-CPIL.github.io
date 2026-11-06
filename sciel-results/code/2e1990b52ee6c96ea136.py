"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct centers of N=100 equal non-overlapping spheres in [-0.5, 0.5]^3
    maximizing the minimum radius.
    
    Approach: Force-directed relaxation (Physics-based).
    We define a potential energy that is high when spheres are close to each other
    or close to the boundaries. Minimizing this energy maximizes the minimum distance.
    
    Potential: U = sum_{i<j} 1/d_ij^2 + sum_{i,k} 1/(dist_to_boundary_ik)^2
    Force: F = -grad(U)
    """
    n = N
    # Initialize with a grid
    # 5x5x4 = 100
    m_x, m_y, m_z = 5, 5, 4
    ticks_x = (np.arange(m_x) + 0.5) / m_x - 0.5
    ticks_y = (np.arange(m_y) + 0.5) / m_y - 0.5
    ticks_z = (np.arange(m_z) + 0.5) / m_z - 0.5
    
    grid = np.array([(x, y, z) for z in ticks_z for y in ticks_y for x in ticks_x])
    centers = grid[:n].copy()
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    centers += rng.uniform(-0.01, 0.01, size=centers.shape)
    
    def compute_r(c):
        """Compute the radius given centers."""
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        x = c[:, 0]
        y = c[:, 1]
        z = c[:, 2]
        boundary_dists = np.min(np.stack([
            0.5 - np.abs(x),
            0.5 - np.abs(y),
            0.5 - np.abs(z)
        ], axis=1), axis=1)
        min_boundary = np.min(boundary_dists)
        
        return min(min_pair_dist / 2.0, min_boundary)
    
    def compute_forces(c):
        """
        Compute forces on all spheres.
        Returns array of shape (N, 3).
        """
        forces = np.zeros_like(c)
        
        # Pairwise repulsion
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(dists_sq, 1.0)
        
        # U = 1/d^2. Force = 2 * diff / d^4
        d4 = dists_sq**2
        
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        
        factor = 2.0 / d4
        factor[~mask] = 0.0
        
        pair_forces = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary repulsion
        # U_b = 1 / (0.5 - abs(x))^2
        # F_x = -2 * sign(x) / (0.5 - abs(x))^3
        
        dist_x = 0.5 - np.abs(c[:, 0])
        dist_x_safe = np.maximum(dist_x, 1e-6)
        sign_x = np.sign(c[:, 0])
        F_x = -2.0 * sign_x / (dist_x_safe**3)
        
        dist_y = 0.5 - np.abs(c[:, 1])
        dist_y_safe = np.maximum(dist_y, 1e-6)
        sign_y = np.sign(c[:, 1])
        F_y = -2.0 * sign_y / (dist_y_safe**3)
        
        dist_z = 0.5 - np.abs(c[:, 2])
        dist_z_safe = np.maximum(dist_z, 1e-6)
        sign_z = np.sign(c[:, 2])
        F_z = -2.0 * sign_z / (dist_z_safe**3)
        
        forces[:, 0] += F_x
        forces[:, 1] += F_y
        forces[:, 2] += F_z
        
        total_forces = pair_forces + forces
        
        # Cap force magnitude to prevent instability
        mag = np.linalg.norm(total_forces, axis=1, keepdims=True)
        mag_safe = np.maximum(mag, 1e-6)
        scale = np.minimum(1.0 / mag_safe, 1.0)
        total_forces = total_forces * scale
        
        return total_forces

    best_centers = centers.copy()
    best_r = compute_r(centers)
    current_r = best_r
    
    # Force-directed relaxation
    num_iters = 5000
    alpha = 0.01
    
    for i in range(num_iters):
        F = compute_forces(centers)
        centers_new = centers + alpha * F
        centers_new = np.clip(centers_new, -0.499, 0.499)
        
        r_new = compute_r(centers_new)
        
        if r_new > current_r:
            centers = centers_new
            current_r = r_new
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            alpha *= 0.95
            
        # Periodic shake to escape local minima
        if i % 1000 == 999:
            centers += rng.uniform(-0.005, 0.005, size=centers.shape)
            centers = np.clip(centers, -0.499, 0.499)
            current_r = compute_r(centers)
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()

    # Final polish: Simple local hill climbing
    for _ in range(10):
        improved = False
        for i in range(n):
            old_center = centers[i].copy()
            for _ in range(10):
                direction = rng.normal(size=3)
                norm = np.linalg.norm(direction)
                if norm < 1e-6: continue
                direction /= norm
                
                for step in [0.001, 0.005, 0.01]:
                    new_center = old_center + step * direction
                    if np.any(np.abs(new_center) > 0.5):
                        continue
                    centers[i] = new_center
                    r_test = compute_r(centers)
                    if r_test > current_r:
                        current_r = r_test
                        improved = True
                        break
                    else:
                        centers[i] = old_center
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
            
    if current_r > best_r:
        best_centers = centers.copy()

    return best_centers
# EVOLVE-BLOCK-END
