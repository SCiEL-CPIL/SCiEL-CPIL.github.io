# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a rigid-body relaxation approach with implicit symplectic integration:
    1. Treat circles as particles with repulsive springs (harmonic potential).
    2. Use implicit symplectic Euler with adaptive time steps for stability.
    3. Binary search on radius r, checking feasibility via relaxation.
    4. Return the best feasible configuration.
    """
    N = 50
    
    def relax_positions(c, r, max_iter=500, dt_init=0.01, damping=0.99):
        """
        Relax positions using implicit symplectic integration.
        Returns (c_final, converged).
        """
        c = c.copy()
        v = np.zeros_like(c)
        
        # Force computation: repulsive springs for overlaps + boundary
        def compute_forces(c, r):
            forces = np.zeros_like(c)
            
            # Pairwise repulsive springs
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Spring force: F = k * max(0, 2r - d) * (diff / d)
            overlap = np.maximum(0, 2 * r - dists)
            safe_dists = np.where(dists > 1e-10, dists, 1.0)
            np.fill_diagonal(overlap, 0)
            
            k = 100.0  # Spring constant
            weights = k * overlap / safe_dists
            np.fill_diagonal(weights, 0)
            
            forces_pairs = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
            forces += forces_pairs
            
            # Boundary force: push inward if outside (1-r)
            radii = np.linalg.norm(c, axis=1)
            bound_violation = np.maximum(0, radii - (1.0 - r))
            safe_radii = np.where(radii > 1e-10, radii, 1.0)
            
            k_boundary = 50.0
            forces_boundary = -k_boundary * bound_violation[:, np.newaxis] * c / safe_radii[:, np.newaxis]
            forces += forces_boundary
            
            return forces
        
        # Implicit symplectic Euler with adaptive time step
        dt = dt_init
        converged = False
        
        for it in range(max_iter):
            F = compute_forces(c, r)
            
            # Check convergence: max force below threshold
            max_force = np.max(np.linalg.norm(F, axis=1))
            if max_force < 1e-8:
                converged = True
                break
            
            # Implicit step: v_new = v + dt * F(v_new)
            # Approximate with semi-implicit: use current F for velocity update
            v = v * damping + dt * F
            
            # Update position
            c = c + dt * v
            
            # Adaptive time step: reduce if forces are large
            if max_force > 1.0:
                dt *= 0.5
            elif max_force < 0.1:
                dt = min(dt * 1.2, 0.05)
            
            # Clamp positions to stay within reasonable bounds
            norms = np.linalg.norm(c, axis=1)
            mask = norms > 1.5
            c[mask] /= norms[mask, np.newaxis] * 1.5
        
        return c, converged
    
    def compute_score(c, r):
        """
        Compute the actual achievable radius for configuration c.
        r_actual = min(min_{i<j} d_ij/2, min_i (1 - |c_i|))
        """
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists) / 2.0
        radii = np.linalg.norm(c, axis=1)
        min_boundary = 1.0 - np.max(radii)
        
        return min(min_pair_dist, min_boundary)
    
    def generate_initial_configurations():
        """
        Generate diverse initial configurations.
        """
        configs = []
        
        # 1. Hexagonal lattice
        for scale in [0.10, 0.12, 0.14, 0.16]:
            points = []
            for j in range(-12, 13):
                for i in range(-12, 13):
                    x = i * scale + 0.5 * scale * (j % 2)
                    y = j * scale * np.sqrt(3) / 2
                    if np.sqrt(x * x + y * y) < 0.95:
                        points.append([x, y])
            points = np.array(points)
            if len(points) >= N:
                dists = np.linalg.norm(points, axis=1)
                indices = np.argsort(dists)[:N]
                configs.append(points[indices])
            else:
                while len(points) < N:
                    angle = np.random.uniform(0, 2 * np.pi)
                    radius = np.random.uniform(0, 0.85)
                    points.append([radius * np.cos(angle), radius * np.sin(angle)])
                configs.append(np.array(points))
        
        # 2. Concentric rings with optimized radii
        # Try multiple ring configurations
        ring_configs = [
            [1, 6, 12, 18, 13],
            [1, 6, 12, 16, 15],
            [1, 6, 10, 14, 19],
            [1, 8, 12, 16, 13],
        ]
        
        for rings in ring_configs:
            total = sum(rings)
            if total != N:
                continue
            
            # Distribute radii to maximize minimum spacing
            radii = []
            for k, n_pts in enumerate(rings):
                if k == 0:
                    radii.append(0.0)
                else:
                    # Approximate optimal ring radius
                    radii.append(0.15 + 0.18 * k)
            
            points = []
            for k, (n_pts, rad) in enumerate(zip(rings, radii)):
                if n_pts == 0:
                    continue
                for i in range(n_pts):
                    angle = 2 * np.pi * i / n_pts + (np.pi / n_pts if k % 2 == 0 else 0)
                    points.append([rad * np.cos(angle), rad * np.sin(angle)])
            if len(points) == N:
                configs.append(np.array(points))
        
        # 3. Random restarts
        for _ in range(8):
            c_rand = np.random.uniform(-0.85, 0.85, (N, 2))
            norms = np.linalg.norm(c_rand, axis=1)
            mask = norms > 0.85
            c_rand[mask] /= norms[mask, np.newaxis] * 0.85
            configs.append(c_rand)
        
        return configs
    
    # Main algorithm
    np.random.seed(42)
    
    configs = generate_initial_configurations()
    
    # Binary search for maximum radius
    r_low = 0.10
    r_high = 0.127  # Slightly below known best
    
    best_centers = None
    best_r = r_low
    
    # Binary search iterations
    for _ in range(25):
        r_mid = (r_low + r_high) / 2
        
        feasible = False
        best_c_for_r = None
        best_score_for_r = -np.inf
        
        for c_init in configs:
            c_opt, converged = relax_positions(c_init, r_mid)
            
            # Compute actual achievable radius
            score = compute_score(c_opt, r_mid)
            
            if score >= r_mid - 1e-9:
                feasible = True
                if score > best_score_for_r:
                    best_score_for_r = score
                    best_c_for_r = c_opt
            elif score > best_score_for_r:
                # Track best even if not fully feasible
                if score > best_score_for_r:
                    best_score_for_r = score
                    best_c_for_r = c_opt
        
        if feasible:
            r_low = r_mid
            best_r = r_mid
            best_centers = best_c_for_r
        else:
            r_high = r_mid
    
    # Final refinement: try to push radius up slightly from best_r
    if best_centers is not None:
        # Try a few extra relaxations at slightly higher radii
        for delta_r in [0.0005, 0.001, 0.002]:
            r_try = best_r + delta_r
            c_try, _ = relax_positions(best_centers, r_try, max_iter=300)
            score = compute_score(c_try, r_try)
            if score >= r_try - 1e-9 and score > best_r:
                best_r = score
                best_centers = c_try
    
    # Final validation and cleanup
    if best_centers is None:
        best_centers = configs[0]
    
    # Ensure all values are finite and within bounds
    best_centers = np.nan_to_num(best_centers, nan=0.0, posinf=0.9, neginf=-0.9)
    norms = np.linalg.norm(best_centers, axis=1)
    mask = norms > 0.99
    best_centers[mask] /= norms[mask, np.newaxis] * 0.99
    
    return best_centers
# EVOLVE-BLOCK-END
