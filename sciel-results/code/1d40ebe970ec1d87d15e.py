# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a physics-based relaxation approach (stochastic gradient descent with cooling)
    to minimize overlap and boundary violations for a fixed radius.
    Binary search is used to find the maximum feasible radius.
    """
    N = 50
    np.random.seed(42)
    
    def energy_and_grad(c, r):
        """
        Compute the penalty energy and its gradient.
        Energy = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, |c_i| - (1-r))^2
        """
        c = c.reshape(N, 2)
        n = N
        energy = 0.0
        grad = np.zeros_like(c)
        
        # Pairwise interactions
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Overlap amount: overlap_ij = max(0, 2r - d_ij)
        overlap = np.maximum(0, 2 * r - dists)
        
        # Energy contribution: sum(overlap^2) / 2
        pair_energy = 0.5 * np.sum(overlap ** 2)
        energy += pair_energy
        
        # Gradient calculation
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        np.fill_diagonal(overlap, 0)
        
        # weights[i,j] = -2 * overlap[i,j] / safe_dists[i,j]
        weights = -2.0 * overlap / safe_dists
        np.fill_diagonal(weights, 0)
        
        grad_pairs = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        grad += grad_pairs
        
        # Boundary interactions
        radii = np.linalg.norm(c, axis=1)
        bound_violation = np.maximum(0, radii - (1.0 - r))
        
        # E_bound = sum (bound_violation)^2
        energy += np.sum(bound_violation ** 2)
        
        # Gradient
        safe_radii = np.where(radii > 1e-10, radii, 1.0)
        grad_bound = 2.0 * bound_violation[:, np.newaxis] * c / safe_radii[:, np.newaxis]
        grad += grad_bound
        
        return energy, grad

    def relax(c_init, r, iterations=500, initial_temp=0.05):
        """
        Relax the configuration using stochastic gradient descent with cooling.
        """
        c = c_init.copy()
        temp = initial_temp
        decay = 0.99
        lr = 0.02
        
        for i in range(iterations):
            e, g = energy_and_grad(c, r)
            
            # Add noise to escape local minima
            if temp > 1e-6:
                noise = np.random.normal(0, temp, c.shape)
            else:
                noise = 0.0
                
            # Gradient step
            c = c - lr * g + noise
            
            # Keep points within a loose bound to prevent flying off
            norms = np.linalg.norm(c, axis=1)
            mask = norms > 1.5
            if np.any(mask):
                c[mask] = c[mask] / norms[mask, np.newaxis] * 1.5
                
            # Cool down
            temp *= decay
            
        e_final, _ = energy_and_grad(c, r)
        return c, e_final

    def generate_initial_configurations():
        """
        Generate a list of diverse initial configurations.
        """
        configs = []
        
        # 1. Hexagonal lattice
        for scale in [0.12, 0.14, 0.16, 0.18]:
            points = []
            for j in range(-10, 11):
                for i in range(-10, 11):
                    x = i * scale + 0.5 * scale * (j % 2)
                    y = j * scale * np.sqrt(3) / 2
                    if np.sqrt(x * x + y * y) < 1.0:
                        points.append([x, y])
            points = np.array(points)
            if len(points) >= N:
                dists = np.linalg.norm(points, axis=1)
                indices = np.argsort(dists)[:N]
                configs.append(points[indices])
            else:
                # Fill remaining with random
                while len(points) < N:
                    angle = np.random.uniform(0, 2 * np.pi)
                    radius = np.random.uniform(0, 0.9)
                    points.append([radius * np.cos(angle), radius * np.sin(angle)])
                configs.append(np.array(points))
        
        # 2. Concentric rings
        rings = [1, 6, 12, 18, 13]
        radii = [0.0, 0.25, 0.5, 0.75, 0.9]
        points = []
        for k, (n_pts, rad) in enumerate(zip(rings, radii)):
            for i in range(n_pts):
                angle = 2 * np.pi * i / n_pts + (np.pi / n_pts if k % 2 == 0 else 0)
                points.append([rad * np.cos(angle), rad * np.sin(angle)])
        configs.append(np.array(points[:N]))
        
        # 3. Random restarts
        for _ in range(10):
            c_rand = np.random.uniform(-0.9, 0.9, (N, 2))
            norms = np.linalg.norm(c_rand, axis=1)
            mask = norms > 0.9
            c_rand[mask] /= norms[mask, np.newaxis] * 0.9
            configs.append(c_rand)
            
        return configs

    # Binary search for the best radius
    r_low = 0.1
    r_high = 0.13
    
    best_centers = None
    best_r = r_low
    
    configs = generate_initial_configurations()
    
    # We will perform a few rounds of binary search
    # To manage time, we use fewer iterations in the binary search loop
    # and more in the final polish.
    
    for _ in range(15):
        r_mid = (r_low + r_high) / 2
        
        feasible = False
        best_c_for_r = None
        best_e_for_r = np.inf
        
        # Try all configs but with fewer iterations
        for c_init in configs:
            c_opt, e_opt = relax(c_init, r_mid, iterations=300)
            if e_opt < best_e_for_r:
                best_e_for_r = e_opt
                best_c_for_r = c_opt
            
            if e_opt < 1e-8:
                feasible = True
                # We can break early if we find a very good feasible solution?
                # No, let's check all to find the "best" feasible one for stability
                # But to save time, if we find one with very low energy, we can stop?
                # Let's just mark feasible and continue to see if we can get lower energy.
                pass
        
        if feasible:
            r_low = r_mid
            best_r = r_mid
            best_centers = best_c_for_r
        else:
            r_high = r_mid
            
    # Final polish at best_r with more iterations
    if best_centers is not None:
        # Use the best config from the last feasible step as initial guess
        c_final, e_final = relax(best_centers, best_r, iterations=1000)
        if e_final < 1e-8:
            best_centers = c_final
        else:
            # Try slightly smaller radius if final polish failed
            r_retry = best_r - 0.0005
            c_retry, e_retry = relax(best_centers, r_retry, iterations=1000)
            if e_retry < 1e-8:
                best_centers = c_retry
                best_r = r_retry

    if best_centers is None:
        # Fallback
        best_centers = configs[0]
        
    return best_centers
# EVOLVE-BLOCK-END
