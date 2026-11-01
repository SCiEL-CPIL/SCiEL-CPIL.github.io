# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a physics-based relaxation approach with an implicit (fixed-point) integrator.
    1. Binary search on radius r.
    2. For each r, relax positions using repulsive forces and boundary constraints.
    3. The integrator uses a large time step with fixed-point iteration (implicit style).
    """
    N = 50
    
    def compute_forces(c, r):
        """
        Compute forces (negative gradient of potential).
        """
        forces = np.zeros_like(c)
        
        # Pairwise repulsion
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Overlap
        overlap = np.maximum(0, 2 * r - dists)
        np.fill_diagonal(overlap, 0)
        
        # Unit vectors
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        u = diff / safe_dists[:, :, np.newaxis]
        
        # Force on i is sum of overlap_ij * u_ij
        # This pushes i away from j if they overlap
        F_pairs = np.sum(overlap[:, :, np.newaxis] * u, axis=1)
        forces += F_pairs
        
        # Boundary
        radii = np.linalg.norm(c, axis=1)
        bound_violation = np.maximum(0, radii - (1.0 - r))
        safe_radii = np.where(radii > 1e-10, radii, 1.0)
        u_bound = c / safe_radii[:, np.newaxis]
        
        # Force pushes inward
        F_bound = -bound_violation[:, np.newaxis] * u_bound
        forces += F_bound
        
        return forces

    def relax(r, c_init, max_iter=100, fixed_point_iter=10, dt=0.5):
        """
        Relax configuration for given radius r.
        Uses implicit-style fixed point iteration.
        """
        c = c_init.copy()
        
        for _ in range(max_iter):
            # Perform fixed point iterations for the implicit step
            # x_new = x_old + dt * F(x_iter)
            x_iter = c.copy()
            for _ in range(fixed_point_iter):
                F = compute_forces(x_iter, r)
                x_iter = c + dt * F
                # Clamp to container to prevent explosion
                # Optional: project onto boundary if violated
                radii = np.linalg.norm(x_iter, axis=1)
                max_r = 1.0 - r + 0.1 # Allow slight violation during iteration
                mask = radii > max_r
                if np.any(mask):
                    x_iter[mask] = x_iter[mask] / radii[mask, np.newaxis] * max_r
            
            c = x_iter
            
            # Check convergence (optional, for speed)
            # If max force is small, break
            F_check = compute_forces(c, r)
            if np.max(np.linalg.norm(F_check, axis=1)) < 1e-6:
                break
                
        return c

    def is_valid(c, r, tol=1e-6):
        """
        Check if configuration is valid for radius r.
        """
        # Boundary
        radii = np.linalg.norm(c, axis=1)
        if np.any(radii > 1.0 - r + tol):
            return False
            
        # Pairwise
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_dist = np.min(dists)
        if min_dist < 2 * r - tol:
            return False
            
        return True

    def generate_initial_config():
        """
        Generate a diverse initial configuration.
        """
        # Concentric rings
        points = []
        # Center
        points.append([0.0, 0.0])
        # Ring 1: 6
        for i in range(6):
            a = 2 * np.pi * i / 6
            points.append([0.25 * np.cos(a), 0.25 * np.sin(a)])
        # Ring 2: 12
        for i in range(12):
            a = 2 * np.pi * i / 12
            points.append([0.5 * np.cos(a), 0.5 * np.sin(a)])
        # Ring 3: 18
        for i in range(18):
            a = 2 * np.pi * i / 18
            points.append([0.75 * np.cos(a), 0.75 * np.sin(a)])
        # Ring 4: 13
        for i in range(13):
            a = 2 * np.pi * i / 13
            points.append([0.95 * np.cos(a), 0.95 * np.sin(a)])
            
        return np.array(points[:N])

    # Binary search
    np.random.seed(42)
    
    r_low = 0.1
    r_high = 0.126 # Known best is ~0.1258
    
    best_centers = None
    best_r = r_low
    c_init = generate_initial_config()
    
    for _ in range(30):
        r_mid = (r_low + r_high) / 2
        
        # Relax
        c_opt = relax(r_mid, c_init)
        
        # Check validity
        if is_valid(c_opt, r_mid):
            r_low = r_mid
            best_r = r_mid
            best_centers = c_opt
            # Update init for next step (warm start)
            c_init = c_opt.copy()
        else:
            r_high = r_mid
            # If failed, maybe try a different init?
            # For now, keep same init, it might just need more relaxation or is truly infeasible.
            # To be safe, if we fail, we might want to restart with a slightly perturbed config
            # but binary search assumes monotonicity. If r is infeasible, smaller r should be feasible
            # if the method is good enough.
            
    # Final polish
    if best_centers is not None:
        # Run more iterations at the best r
        best_centers = relax(best_r, best_centers, max_iter=500, fixed_point_iter=20)
        
    if best_centers is None:
        best_centers = c_init
        
    return best_centers
# EVOLVE-BLOCK-END
