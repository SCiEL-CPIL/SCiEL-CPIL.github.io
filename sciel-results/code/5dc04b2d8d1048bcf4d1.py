# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp

def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a physics-based relaxation approach:
    1. Model circles as particles with repulsive forces when overlapping and boundary forces.
    2. Use an implicit symplectic-like integrator (Radau from SciPy) for stability.
    3. Binary search for the maximum radius r where the system reaches a stable, non-overlapping state.
    """
    N = 50
    # Parameters for the dynamics
    k_rep = 100.0   # Stiffness for overlap repulsion
    k_bound = 100.0 # Stiffness for boundary repulsion
    damping = 1.0   # Damping coefficient (viscosity)
    dt_max = 0.1    # Max time step for integrator
    t_end = 50.0    # Total simulation time
    
    def get_forces(centers, r):
        """
        Calculate forces on each particle.
        centers: (N, 2) array
        r: radius
        Returns: (N, 2) array of forces
        """
        forces = np.zeros_like(centers)
        n = N
        
        # Pairwise repulsion
        # Vectorized distance calculation
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Overlap: 2r - d
        overlap = 2 * r - dists
        # Only apply force if overlapping
        mask = overlap > 0
        # Force magnitude: k * overlap
        # Direction: normalized diff
        # F_ij on i is -k * overlap_ij * (c_i - c_j) / d_ij
        # Note: diff[i,j] = c_i - c_j. So direction from j to i is diff[i,j]/d.
        # Force on i should push it away from j. So +k * overlap * diff[i,j] / d.
        
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        np.fill_diagonal(safe_dists, 1.0)
        
        force_mags = np.where(mask, k_rep * overlap, 0.0)
        # Force vector contribution from j to i
        # F_i = sum_j (k * overlap_ij / d_ij) * (c_i - c_j)
        # term = force_mags[:, :, np.newaxis] / safe_dists[:, :, np.newaxis] * diff
        forces += np.sum(force_mags[:, :, np.newaxis] / safe_dists[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary repulsion
        radii = np.linalg.norm(centers, axis=1)
        bound_overlap = radii - (1.0 - r)
        bound_mask = bound_overlap > 0
        safe_radii = np.where(radii > 1e-10, radii, 1.0)
        
        # Force towards center: -k * bound_overlap * (c_i / |c_i|)
        bound_force_mags = np.where(bound_mask, -k_bound * bound_overlap, 0.0)
        forces += bound_force_mags[:, np.newaxis] * centers / safe_radii[:, np.newaxis]
        
        return forces

    def simulate(centers_init, r):
        """
        Simulate the system for a fixed radius r.
        Returns the final centers.
        """
        c_flat = centers_init.flatten()
        
        def dynamics(t, y):
            c = y.reshape(N, 2)
            f = get_forces(c, r)
            # Damping: -damping * v. But we don't track v explicitly in y?
            # Standard ODE: m*a = F. With damping: m*a = F - c*v.
            # To use solve_ivp, we need state [pos, vel].
            # Let's redefine state to include velocity.
            pass # Placeholder, need to change state vector
            
        # State vector: [x1, y1, v1x, v1y, x2, y2, v2x, v2y, ...]
        # Size: 4N
        y0 = np.zeros(4 * N)
        y0[0::4] = centers_init[:, 0]
        y0[1::4] = centers_init[:, 1]
        # Initial velocity 0
        
        def dynamics(t, y):
            pos = y[0::4].reshape(N, 2)
            vel = y[1::4].reshape(N, 2)
            
            forces = get_forces(pos, r)
            
            # a = (F - damping * v) / m (m=1)
            acc = forces - damping * vel
            
            dy = np.zeros_like(y)
            dy[0::4] = vel
            dy[1::4] = acc
            return dy
        
        try:
            sol = solve_ivp(dynamics, [0, t_end], y0, method='Radau', 
                            max_step=dt_max, rtol=1e-6, atol=1e-9)
            if sol.success:
                final_y = sol.y[:, -1]
                final_centers = np.column_stack((final_y[0::4], final_y[1::4]))
                return final_centers
            else:
                return centers_init
        except Exception:
            return centers_init

    def check_validity(centers, r):
        """
        Check if the configuration is valid (no overlaps, inside boundary).
        Returns min margin (positive if valid).
        """
        c = centers
        # Pairwise
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        margin_pair = min_pair_dist - 2 * r
        
        # Boundary
        radii = np.linalg.norm(c, axis=1)
        max_radius = np.max(radii)
        margin_bound = (1.0 - r) - max_radius
        
        return min(margin_pair, margin_bound)

    def generate_initial_configs():
        configs = []
        # 1. Fermat Spiral
        golden_angle = np.pi * (3 - np.sqrt(5))
        points = []
        for i in range(N):
            r = 0.9 * np.sqrt(i / N)
            theta = i * golden_angle
            points.append([r * np.cos(theta), r * np.sin(theta)])
        configs.append(np.array(points))
        
        # 2. Concentric Rings (approx)
        rings = [1, 6, 12, 18, 13]
        radii = [0.0, 0.2, 0.4, 0.6, 0.8]
        points = []
        for k, (n_pts, rad) in enumerate(zip(rings, radii)):
            for i in range(n_pts):
                angle = 2 * np.pi * i / n_pts + (np.pi / n_pts if k % 2 == 0 else 0)
                points.append([rad * np.cos(angle), rad * np.sin(angle)])
        configs.append(np.array(points[:N]))
        
        # 3. Random
        for _ in range(3):
            c_rand = np.random.uniform(-0.8, 0.8, (N, 2))
            norms = np.linalg.norm(c_rand, axis=1)
            mask = norms > 0.8
            c_rand[mask] /= norms[mask, np.newaxis] * 0.8
            configs.append(c_rand)
            
        return configs

    np.random.seed(42)
    configs = generate_initial_configs()
    
    r_low = 0.1
    r_high = 0.13
    
    best_centers = None
    best_r = r_low
    
    # Binary search
    for _ in range(15):
        r_mid = (r_low + r_high) / 2
        
        feasible = False
        best_margin = -np.inf
        best_c = None
        
        for c_init in configs:
            c_final = simulate(c_init, r_mid)
            margin = check_validity(c_final, r_mid)
            
            if margin > best_margin:
                best_margin = margin
                best_c = c_final
                
            if margin > 1e-4:
                feasible = True
                break # Found a valid config for this r, move on
        
        if feasible:
            r_low = r_mid
            best_r = r_mid
            best_centers = best_c
        else:
            r_high = r_mid
            # If not feasible, keep the best margin config for next iteration's init?
            # Or just discard.
            
    # Final polish: use the best centers found at r_low, and try to optimize slightly higher?
    # Or just return best_centers.
    # To improve score, we can try to "push" the radius a bit higher with the best config.
    if best_centers is not None:
        # Try a few steps up from r_low
        for delta in [0.001, 0.0005, 0.0002]:
            r_try = r_low + delta
            c_try = simulate(best_centers, r_try)
            margin = check_validity(c_try, r_try)
            if margin > -1e-5: # Allow tiny numerical errors
                best_r = r_try
                best_centers = c_try
                break

    if best_centers is None:
        best_centers = configs[0]
        
    return best_centers
# EVOLVE-BLOCK-END
