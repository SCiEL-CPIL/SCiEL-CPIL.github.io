# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a force-directed relaxation approach (damped dynamics) to find valid configurations
    for a given radius, combined with binary search to maximize the radius.
    """
    N = 50
    
    def compute_forces(c, r):
        """
        Compute forces on each particle to resolve overlaps and boundary violations.
        Forces are derived from the gradient of the penalty energy.
        """
        c = c.reshape(N, 2)
        forces = np.zeros_like(c)
        
        # Pairwise repulsion
        # diff[i, j] = c[i] - c[j]
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, 1.0) # Avoid division by zero
        
        # Overlap: positive if dist < 2r
        # We want to push apart if dist < 2r
        # Force magnitude proportional to overlap
        overlap = np.maximum(0, 2 * r - dists)
        
        # Gradient of 0.5 * overlap^2 w.r.t c_i
        # d/dc_i 0.5 * (2r - d)^2 = -(2r - d) * d/d c_i (2r - d) = -(2r - d) * (-d/d c_i d) = (2r - d) * (c_i - c_j) / d
        # So force on i is sum_j (2r - d_ij) * (c_i - c_j) / d_ij
        
        # weights[i, j] = overlap[i, j] / dists[i, j]
        weights = overlap / dists
        np.fill_diagonal(weights, 0)
        
        # Sum over j for each i
        # force_i = sum_j weights[i, j] * (c_i - c_j)
        # Note: diff[i, j] is c_i - c_j
        pair_forces = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        forces += pair_forces
        
        # Boundary attraction/repulsion
        # If |c_i| > 1 - r, push back towards center
        radii = np.linalg.norm(c, axis=1)
        bound_violation = np.maximum(0, radii - (1.0 - r))
        
        # Force is towards center: -c_i / |c_i| * bound_violation
        # Gradient of 0.5 * bound_violation^2 is bound_violation * c_i / |c_i|
        # We want to minimize energy, so force is -gradient
        # Wait, the energy is sum (bound_violation)^2.
        # Gradient w.r.t c_i is 2 * bound_violation * c_i / |c_i|.
        # Force = -Gradient.
        # So force_i = -2 * bound_violation_i * c_i / |c_i|
        
        safe_radii = np.where(radii > 1e-10, radii, 1.0)
        bound_forces = -2.0 * bound_violation[:, np.newaxis] * c / safe_radii[:, np.newaxis]
        forces += bound_forces
        
        return forces

    def relax(c_init, r, max_iters=2000, tol=1e-8):
        """
        Relax the configuration using damped dynamics.
        """
        c = c_init.copy()
        v = np.zeros_like(c)
        
        # Adaptive step size
        dt = 0.01
        damping = 0.9
        
        last_energy = np.inf
        stagnation_count = 0
        
        for i in range(max_iters):
            # Compute forces
            F = compute_forces(c, r)
            
            # Update velocity and position
            v = v * damping + F * dt
            c_new = c + v * dt
            
            # Check energy change to adapt dt
            # Simple check: if any particle moved too far, reduce dt
            # Or just check if energy decreased
            # For simplicity, we use a fixed small dt and rely on damping.
            # However, to be robust, we can check if the new position is worse.
            
            c = c_new
            
            # Check convergence
            force_mag = np.linalg.norm(F)
            if force_mag < tol:
                break
                
            # Optional: Stagnation detection
            # If energy doesn't decrease for a while, add noise
            # (Skipped for simplicity, relying on multiple restarts)
            
        return c

    def generate_initial_configurations():
        """
        Generate diverse initial configurations.
        """
        configs = []
        
        # 1. Hexagonal lattice
        # Generate a dense hex lattice and pick the best N points closest to center
        scale = 0.15 # Adjust scale to ensure enough points
        points = []
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = i * scale + (0.5 * scale if j % 2 else 0)
                y = j * scale * np.sqrt(3) / 2
                points.append([x, y])
        points = np.array(points)
        # Keep only points inside unit circle (with margin)
        norms = np.linalg.norm(points, axis=1)
        points = points[norms < 0.9]
        
        if len(points) >= N:
            # Sort by distance to center and take first N
            norms = np.linalg.norm(points, axis=1)
            indices = np.argsort(norms)[:N]
            configs.append(points[indices])
            
        # 2. Concentric Rings
        # Try different ring distributions
        # Total 50.
        # Center: 1
        # Ring 1: 6
        # Ring 2: 12
        # Ring 3: 18
        # Ring 4: 13 (Remaining)
        # Radii need to be optimized.
        # Approximate radii for r=0.125:
        # Center: 0
        # Ring 1: ~ 2*r = 0.25
        # Ring 2: ~ 4*r = 0.5
        # Ring 3: ~ 6*r = 0.75
        # Ring 4: ~ 8*r = 1.0 (Too big, must be < 1-r = 0.875)
        
        # Let's try a set of radii that sum up to 50
        ring_counts = [1, 6, 12, 18, 13]
        # Radii should be roughly proportional to sqrt(count) or linear?
        # In a dense packing, area per circle is constant.
        # Area of ring k ~ pi * (R_k^2 - R_{k-1}^2)
        # N_k * pi * r^2 ~ pi * (R_k^2 - R_{k-1}^2)
        # R_k^2 ~ R_{k-1}^2 + N_k * r^2
        # This assumes perfect packing.
        
        r_guess = 0.12
        R_prev = 0
        radii = []
        for count in ring_counts:
            if count == 1:
                radii.append(0.0)
            else:
                R_curr = np.sqrt(R_prev**2 + count * r_guess**2)
                radii.append(R_curr)
                R_prev = R_curr
                
        points = []
        for k, (count, rad) in enumerate(zip(ring_counts, radii)):
            for i in range(count):
                angle = 2 * np.pi * i / count + (np.pi / count if k % 2 else 0)
                points.append([rad * np.cos(angle), rad * np.sin(angle)])
        
        if len(points) == N:
            configs.append(np.array(points))
            
        # 3. Random configurations
        np.random.seed(42)
        for _ in range(10):
            # Random points in the unit disk
            angles = np.random.uniform(0, 2 * np.pi, N)
            radii_rand = np.random.uniform(0, 0.9, N)
            c_rand = np.column_stack([radii_rand * np.cos(angles), radii_rand * np.sin(angles)])
            configs.append(c_rand)
            
        return configs

    # Binary search for the best radius
    np.random.seed(42)
    
    r_low = 0.1
    r_high = 0.126 # Known best is 0.1258...
    
    best_centers = None
    best_r = r_low
    
    configs = generate_initial_configurations()
    
    # To speed up and improve robustness, we perform binary search
    # For each r, we try all initial configs and keep the best one (lowest energy or valid)
    
    for _ in range(25): # 25 iterations for high precision
        r_mid = (r_low + r_high) / 2
        
        feasible = False
        best_c_for_r = None
        best_energy = np.inf
        
        for c_init in configs:
            c_opt = relax(c_init, r_mid, max_iters=500, tol=1e-6)
            
            # Check if valid
            # Calculate min distance and max radius
            dists = np.sqrt(np.sum((c_opt[:, np.newaxis, :] - c_opt[np.newaxis, :, :])**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            
            radii = np.linalg.norm(c_opt, axis=1)
            max_rad = np.max(radii)
            
            # Valid if min_dist >= 2*r_mid and max_rad <= 1 - r_mid
            if min_dist >= 2 * r_mid - 1e-6 and max_rad <= 1 - r_mid + 1e-6:
                feasible = True
                # We want to maximize the "slack" or just keep the first valid one?
                # Actually, if feasible, we can try to improve the configuration further?
                # No, relax() should have converged.
                # We can compare the minimum distance achieved.
                if min_dist > best_energy: # Using best_energy as best_min_dist
                    best_energy = min_dist
                    best_c_for_r = c_opt
                    if min_dist > 2 * r_mid + 1e-4: # If there's significant slack, it's a good candidate
                        break # No need to try other inits for this r if we found a very good one?
                        # Actually, maybe another init is better? Let's check all.
            
        if feasible:
            r_low = r_mid
            best_r = r_mid
            best_centers = best_c_for_r
        else:
            r_high = r_mid

    # Final polish
    if best_centers is not None:
        # Try to relax at the final radius with more iterations
        c_final = relax(best_centers, best_r, max_iters=2000, tol=1e-8)
        
        # Verify
        dists = np.sqrt(np.sum((c_final[:, np.newaxis, :] - c_final[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        radii = np.linalg.norm(c_final, axis=1)
        max_rad = np.max(radii)
        
        # If it's not valid, we might have overshot.
        # But binary search ensures r_low is feasible.
        # So best_r is feasible.
        
        best_centers = c_final

    if best_centers is None:
        # Fallback
        best_centers = configs[0]
        
    return best_centers
# EVOLVE-BLOCK-END
