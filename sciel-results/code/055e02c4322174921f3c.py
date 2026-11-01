# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a physics-based relaxation approach:
    1. Model circles as particles with repulsive harmonic springs when overlapping.
    2. Apply boundary forces to keep circles inside the container.
    3. Use a Velocity Verlet integrator with friction (Langevin dynamics) for stability.
    4. Iteratively increase the target radius r until the system cannot maintain a valid configuration.
    """
    N = 50
    # Parameters for the physics simulation
    k_rep = 100.0      # Repulsive spring constant
    k_bound = 100.0    # Boundary spring constant
    friction = 0.9     # Damping factor for velocity (Langevin)
    dt = 0.01          # Time step
    n_steps = 500      # Number of simulation steps per radius check
    temp = 0.01        # Initial temperature for Langevin noise (optional, set to 0 for deterministic)
    
    def simulate(c, v, r, steps, k_rep, k_bound, friction, dt, temp=0.0):
        """
        Run the physics simulation for a fixed number of steps.
        Returns final positions and velocities.
        """
        c = c.copy()
        v = v.copy()
        
        for step in range(steps):
            # Calculate forces
            F = np.zeros_like(c)
            
            # Pairwise repulsive forces
            # Vectorized calculation of pairwise distances
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Overlap condition: dist < 2r
            overlap_mask = dists < 2.0 * r
            # Force magnitude: k_rep * (2r - dist)
            # We only care about pairs where dist < 2r
            # To avoid division by zero and handle the mask efficiently:
            # Force direction is along the vector from j to i (c_i - c_j)
            # F_i = sum_j k_rep * (2r - d_ij) * (c_i - c_j) / d_ij
            
            # Create a safe distance matrix for division
            safe_dists = np.where(dists > 1e-10, dists, 1.0)
            
            # Compute force components
            # term_ij = k_rep * max(0, 2r - d_ij) / d_ij
            # We use the mask to zero out non-overlapping pairs
            force_weights = np.zeros_like(dists)
            force_weights[overlap_mask] = k_rep * (2.0 * r - dists[overlap_mask]) / safe_dists[overlap_mask]
            
            # Sum forces: F_i = sum_j force_weights[i,j] * (c_i - c_j)
            # Note: diff[i,j] = c_i - c_j
            F += np.sum(force_weights[:, :, np.newaxis] * diff, axis=1)
            
            # Boundary forces
            radii = np.linalg.norm(c, axis=1)
            bound_violation = np.maximum(0, radii - (1.0 - r))
            # Force is inward: -k_bound * violation * (c_i / |c_i|)
            safe_radii = np.where(radii > 1e-10, radii, 1.0)
            F_bound = -k_bound * bound_violation[:, np.newaxis] * c / safe_radii[:, np.newaxis]
            F += F_bound
            
            # Integrate using Velocity Verlet with friction
            # v_new = friction * v + F * dt
            # c_new = c + v_new * dt
            # Note: Standard Verlet uses half steps, but with friction/damping, 
            # a simple Euler or semi-implicit Euler is often more stable for packing.
            # Let's use Semi-Implicit Euler (Symplectic Euler) which is stable for Hamiltonian systems with damping.
            v = friction * v + F * dt
            c = c + v * dt
            
            # Add thermal noise if temp > 0 (Langevin)
            if temp > 0:
                v += np.random.normal(0, temp, c.shape)
                
            # Optional: Hard constraint projection to prevent extreme excursions
            # If a circle goes far outside, pull it back gently
            radii = np.linalg.norm(c, axis=1)
            max_r = 1.0 + r # Allow some slack
            mask_out = radii > max_r
            if np.any(mask_out):
                c[mask_out] /= radii[mask_out, np.newaxis] * max_r
                v[mask_out] = 0 # Dampen velocity of escaping particles
        
        return c, v

    def is_valid(c, r, tol=1e-6):
        """
        Check if the configuration is valid for radius r.
        """
        # Check boundary
        radii = np.linalg.norm(c, axis=1)
        if np.any(radii > 1.0 - r + tol):
            return False
        
        # Check overlaps
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        if np.any(dists < 2.0 * r - tol):
            return False
            
        return True

    def solve_for_radius(r, c_init, v_init=None):
        """
        Attempt to find a valid configuration for radius r.
        Returns (c_opt, success).
        """
        if v_init is None:
            v_init = np.zeros_like(c_init)
        
        c_curr = c_init.copy()
        v_curr = v_init.copy()
        
        # Run simulation
        c_curr, v_curr = simulate(c_curr, v_curr, r, n_steps, k_rep, k_bound, friction, dt, temp=0.0)
        
        # Check validity
        if is_valid(c_curr, r):
            return c_curr, True
        
        # If failed, try a few more steps with slightly different parameters or just return failure
        # To save time, we don't do extensive retry logic here, but we can try once more with higher friction
        c_curr2, v_curr2 = simulate(c_curr, v_curr, r, n_steps//2, k_rep, k_bound, 0.95, dt, temp=0.0)
        if is_valid(c_curr2, r):
            return c_curr2, True
            
        return c_curr, False

    # Initial configuration
    # Generate a hexagonal lattice pattern
    def make_initial_hexagonal(n, scale=0.15):
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

    np.random.seed(42)
    
    # Start with a small radius and increase
    r = 0.05
    c_current = make_initial_hexagonal(N, scale=0.15)
    v_current = np.zeros_like(c_current)
    
    best_r = -np.inf
    best_centers = c_current.copy()
    
    # Increase radius in small steps
    # We use a binary-search-like strategy or linear increase with backtracking
    # Linear increase is simpler and often works well with physics relaxation
    step_size = 0.002
    
    for i in range(100): # Max 100 steps
        c_opt, success = solve_for_radius(r, c_current, v_current)
        
        if success:
            # Update current state
            c_current = c_opt
            v_current = np.zeros_like(c_current) # Reset velocity to help next step
            
            # Record best
            if r > best_r:
                best_r = r
                best_centers = c_opt.copy()
                
            # Increase radius
            r += step_size
        else:
            # Failed to find valid config for current r
            # Try to refine by decreasing step size or breaking
            # For simplicity, we break and use the last valid r
            # We could try r - step_size/2, but let's keep it simple
            break
            
    # Final polish: try to optimize the best found configuration a bit more
    # We can run the simulation again with the best r to ensure stability
    if best_r > 0:
        c_polished, v_polished = simulate(best_centers, np.zeros_like(best_centers), best_r, n_steps, k_rep, k_bound, friction, dt, 0.0)
        if is_valid(c_polished, best_r):
            best_centers = c_polished

    return best_centers
# EVOLVE-BLOCK-END
