# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a physics-based relaxation approach:
    1. Treat circles as particles with repulsive interactions.
    2. Use a symplectic integrator (Velocity Verlet with damping) to relax the system.
    3. Gradually increase the target radius r, allowing the system to find a valid configuration.
    """
    N = 50
    
    # Initial configuration: Hexagonal lattice scaled to fit
    def generate_initial_config():
        points = []
        # Generate a grid of points in a hexagonal pattern
        for i in range(-10, 11):
            for j in range(-10, 11):
                x = i * 0.2
                y = j * 0.2 * np.sqrt(3)
                if i % 2 != 0:
                    y += 0.1 * np.sqrt(3)
                if np.sqrt(x*x + y*y) < 0.8:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) >= N:
            # Select N points closest to center
            dists = np.linalg.norm(points, axis=1)
            indices = np.argsort(dists)[:N]
            return points[indices]
        else:
            # Fill with random if not enough (should not happen with above params)
            while len(points) < N:
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0, 0.5)
                points.append([radius * np.cos(angle), radius * np.sin(angle)])
            return np.array(points[:N])

    # Physics simulation parameters
    dt = 0.01
    damping = 0.9
    stiffness = 100.0
    max_iters = 2000
    force_tol = 1e-5

    def simulate(c, r, steps):
        """
        Run physics simulation to relax the system for a given radius r.
        c: (N, 2) array of centers.
        """
        c = c.copy()
        v = np.zeros_like(c)
        
        for _ in range(steps):
            # Calculate forces
            forces = np.zeros_like(c)
            
            # Pairwise repulsion
            # Vectorized distance calculation
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Overlap amount: max(0, 2r - d)
            overlaps = np.maximum(0, 2 * r - dists)
            
            # Force magnitude: stiffness * overlap
            # Direction: normalized diff
            # F_ij = stiffness * overlap_ij * (c_i - c_j) / d_ij
            # We sum forces on i from all j
            
            # To avoid division by zero, use safe dists
            safe_dists = np.where(dists > 1e-8, dists, 1.0)
            
            # Weight for each pair
            weights = stiffness * overlaps / safe_dists
            np.fill_diagonal(weights, 0)
            
            # Sum forces
            # force_i = sum_j weights[i,j] * (c_i - c_j)
            # diff[i,j] = c_i - c_j
            forces += np.sum(weights[:, :, np.newaxis] * diff, axis=1)
            
            # Boundary forces
            radii = np.linalg.norm(c, axis=1)
            bound_violations = np.maximum(0, radii - (1.0 - r))
            safe_radii = np.where(radii > 1e-8, radii, 1.0)
            
            # Force pushing inward: stiffness * violation * (c / |c|)
            # Direction is -c/|c|, so force is -stiffness * violation * c/|c|
            # Wait, if c is outside, we want to push it back to origin.
            # Force = -stiffness * bound_violation * (c / |c|)
            forces -= stiffness * bound_violations[:, np.newaxis] * (c / safe_radii[:, np.newaxis])
            
            # Update velocities (Symplectic Euler / Semi-implicit)
            v += forces * dt
            v *= damping
            
            # Update positions
            c += v * dt
            
            # Check convergence
            if np.max(np.linalg.norm(forces, axis=1)) < force_tol:
                break
                
        return c

    # Main logic: Binary search or linear increase of r
    # We know the best is ~0.1258. Let's try to find the max feasible r.
    # Strategy: Start with small r, increase until relaxation fails (energy doesn't go to 0).
    
    np.random.seed(42)
    c = generate_initial_config()
    
    # Start with a safe radius
    r_current = 0.1
    r_max = 0.126 # Upper bound estimate
    
    # First, relax at r_current to get a good starting point
    c = simulate(c, r_current, max_iters)
    
    best_c = c.copy()
    best_r = r_current
    
    # Incremental increase
    step_size = 0.0005
    r_try = r_current + step_size
    
    while r_try <= r_max + 0.001:
        # Try to relax at r_try
        c_new = simulate(c, r_try, max_iters)
        
        # Check if configuration is valid (minimal overlap and boundary violation)
        dists = np.sqrt(np.sum((c_new[:, np.newaxis, :] - c_new[np.newaxis, :, :]) ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        radii = np.linalg.norm(c_new, axis=1)
        max_radius = np.max(radii)
        
        # Valid if min_dist >= 2*r_try and max_radius <= 1 - r_try
        # Allow small tolerance
        tol = 1e-4
        if min_dist >= 2 * r_try - tol and max_radius <= 1 - r_try + tol:
            # Success
            best_c = c_new.copy()
            best_r = r_try
            c = c_new
            r_try += step_size
        else:
            # Failure, try to recover or stop
            # If it fails, we can't go higher. 
            # We might try a smaller step or break.
            # For robustness, if we fail, we break and return the last valid one.
            # But maybe we can do a local search? 
            # Let's just break for now, as the directive implies a single relaxation pass.
            break
            
    return best_c
# EVOLVE-BLOCK-END
