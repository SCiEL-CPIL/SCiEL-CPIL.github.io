# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a physics-based relaxation approach (Molecular Dynamics) with 
    repulsive springs for overlaps and boundary violations.
    1. Define a potential energy with stiff repulsive forces.
    2. Use a Velocity Verlet integrator with damping to relax the system.
    3. Binary search for the maximum radius r such that a valid configuration exists.
    """
    N = 50
    
    def compute_forces(c, r):
        """
        Compute forces on particles.
        F = -grad E
        E = sum_{i<j} 0.5 * k * max(0, 2r - d_ij)^2 + sum_i 0.5 * k_b * max(0, |c_i| - (1-r))^2
        """
        c = c.reshape(N, 2)
        n = N
        forces = np.zeros_like(c)
        
        # Pairwise interactions
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Overlap amount: overlap_ij = max(0, 2r - d_ij)
        overlap = np.maximum(0, 2 * r - dists)
        
        # Force magnitude: F = k * overlap
        # Direction: diff / dist
        # F_ij = k * overlap * (c_i - c_j) / d_ij
        # Note: If d_ij is 0, force is undefined, but overlap is 2r. 
        # We handle small distances carefully.
        
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        # Unit vector from j to i
        u = diff / safe_dists[:, :, np.newaxis]
        
        # Force on i due to j: k * overlap_ij * u_ij
        # Sum over j
        k_spring = 100.0 # Stiff spring constant
        forces += k_spring * np.sum(overlap[:, :, np.newaxis] * u, axis=1)
        
        # Boundary interactions
        radii = np.linalg.norm(c, axis=1)
        bound_violation = np.maximum(0, radii - (1.0 - r))
        
        # Force pushing back to center
        # F = -k_b * violation * (c / |c|)
        safe_radii = np.where(radii > 1e-10, radii, 1.0)
        k_boundary = 100.0
        forces -= k_boundary * bound_violation[:, np.newaxis] * c / safe_radii[:, np.newaxis]
        
        return forces

    def md_relax(c_init, r, steps=1000, dt=0.01, damping=0.9):
        """
        Relax the system using Velocity Verlet with damping.
        """
        c = c_init.copy()
        v = np.zeros_like(c)
        
        f = compute_forces(c, r)
        
        for _ in range(steps):
            # Update positions
            c += v * dt + 0.5 * f * dt * dt
            
            # Compute new forces
            f_new = compute_forces(c, r)
            
            # Update velocities (with damping)
            v = (v + 0.5 * (f + f_new) * dt) * damping
            f = f_new
            
        # Calculate final energy to check feasibility
        # We need a function to calculate energy, similar to forces but integrated
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        overlap = np.maximum(0, 2 * r - dists)
        pair_energy = 0.5 * np.sum(overlap ** 2)
        
        radii = np.linalg.norm(c, axis=1)
        bound_violation = np.maximum(0, radii - (1.0 - r))
        bound_energy = 0.5 * np.sum(bound_violation ** 2)
        
        total_energy = pair_energy + bound_energy
        return c, total_energy

    def solve_for_radius(r, c_init, md_steps=500):
        """
        Relax configuration for fixed radius r.
        Returns (c_opt, min_energy).
        """
        c_opt, e_opt = md_relax(c_init, r, steps=md_steps)
        return c_opt, e_opt

    def generate_initial_configurations():
        """
        Generate a list of diverse initial configurations.
        """
        configs = []
        
        # 1. Random uniform
        for _ in range(10):
            c_rand = np.random.uniform(-0.9, 0.9, (N, 2))
            norms = np.linalg.norm(c_rand, axis=1)
            mask = norms > 0.9
            c_rand[mask] /= norms[mask, np.newaxis] * 0.9
            configs.append(c_rand)
            
        # 2. Concentric rings
        rings = [1, 6, 12, 18, 13]
        radii = [0.0, 0.25, 0.5, 0.75, 0.9]
        points = []
        for k, (n_pts, rad) in enumerate(zip(rings, radii)):
            for i in range(n_pts):
                angle = 2 * np.pi * i / n_pts + (np.pi / n_pts if k % 2 == 0 else 0)
                points.append([rad * np.cos(angle), rad * np.sin(angle)])
        configs.append(np.array(points[:N]))
        
        return configs

    # Binary search for the best radius
    np.random.seed(42)
    
    r_low = 0.1
    r_high = 0.13
    
    best_centers = None
    best_r = r_low
    
    configs = generate_initial_configurations()
    
    # Binary search iterations
    # We reduce the number of MD steps as we get closer to the optimum to save time?
    # Or keep it constant. 120s limit.
    # 20 iterations * 11 configs * 500 steps * (N^2 ops)
    # N=50, N^2=2500. 500 * 2500 = 1.25M ops per config.
    # 11 * 1.25M = 13.75M ops per binary search step.
    # 20 * 13.75M = 275M ops. Should be fast in Python/NumPy.
    
    for _ in range(20):
        r_mid = (r_low + r_high) / 2
        
        feasible = False
        best_c_for_r = None
        best_e_for_r = np.inf
        
        for c_init in configs:
            c_opt, e_opt = solve_for_radius(r_mid, c_init)
            if e_opt < best_e_for_r:
                best_e_for_r = e_opt
                best_c_for_r = c_opt
            
            if e_opt < 1e-6:
                feasible = True
        
        if feasible:
            r_low = r_mid
            best_r = r_mid
            best_centers = best_c_for_r
        else:
            r_high = r_mid
            
    # Final polish at best_r
    if best_centers is not None:
        # Run more steps to ensure convergence
        c_final, e_final = solve_for_radius(best_r, best_centers, md_steps=2000)
        if e_final < 1e-6:
            best_centers = c_final
        else:
            # If final polish failed, try slightly smaller radius
            r_retry = best_r - 0.001
            c_retry, e_retry = solve_for_radius(r_retry, best_centers, md_steps=2000)
            if e_retry < 1e-6:
                best_centers = c_retry
                best_r = r_retry

    if best_centers is None:
        # Fallback
        best_centers = configs[0]
        
    return best_centers
# EVOLVE-BLOCK-END
