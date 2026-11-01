# EVOLVE-BLOCK-START
import numpy as np

def construct():
    N = 50
    
    def compute_forces_and_energy(c, r):
        """
        Compute forces (negative gradient of energy) and energy.
        Energy = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, |c_i| - (1-r))^2
        """
        c = c.reshape(N, 2)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, 1.0) # Avoid div by 0
        
        # Pairwise
        overlap = np.maximum(0, 2 * r - dists)
        # Force is -grad E. 
        # E_pair = sum 0.5 * overlap^2 (if we count i<j once)
        # Let's use full matrix and divide by 2 at the end for energy, but forces sum up.
        # grad_i = sum_j -2 * overlap_ij * (c_i - c_j) / d_ij
        # Force_i = -grad_i = sum_j 2 * overlap_ij * (c_i - c_j) / d_ij
        
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        np.fill_diagonal(overlap, 0)
        
        weights = 2.0 * overlap / safe_dists
        np.fill_diagonal(weights, 0)
        
        forces_pairs = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary
        radii = np.linalg.norm(c, axis=1)
        bound_viol = np.maximum(0, radii - (1.0 - r))
        
        # E_bound = sum bound_viol^2
        # grad_i = 2 * bound_viol_i * c_i / radii_i
        # Force_i = -grad_i = -2 * bound_viol_i * c_i / radii_i
        
        safe_radii = np.where(radii > 1e-10, radii, 1.0)
        forces_bound = -2.0 * bound_viol[:, np.newaxis] * c / safe_radii[:, np.newaxis]
        
        total_forces = forces_pairs + forces_bound
        
        energy = 0.5 * np.sum(overlap ** 2) + np.sum(bound_viol ** 2)
        
        return energy, total_forces

    def relax(c_init, r, max_iters=500):
        c = c_init.copy()
        v = np.zeros_like(c) # Momentum
        alpha = 0.1 # Step size
        beta = 0.9 # Momentum decay
        
        for i in range(max_iters):
            e, f = compute_forces_and_energy(c, r)
            
            if e < 1e-10:
                break
                
            # Update velocity
            v = beta * v + alpha * f
            
            # Tentative step
            c_new = c + v
            
            # Check if energy decreased (simple line search)
            e_new, _ = compute_forces_and_energy(c_new, r)
            
            if e_new < e:
                c = c_new
            else:
                # Reduce step size
                alpha *= 0.5
                v = beta * v + alpha * f
                c_new = c + v
                e_new, _ = compute_forces_and_energy(c_new, r)
                if e_new < e:
                    c = c_new
                else:
                    # If still failing, keep c and reduce alpha further
                    alpha *= 0.5
                    v = 0.5 * v
            
        return c

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
    
    best_centers = None
    best_r = -np.inf
    
    # Try multiple initial configurations
    initial_configs = []
    for scale in [0.12, 0.14, 0.16]:
        initial_configs.append(make_initial_hexagonal(N, scale=scale))
        
    for _ in range(5):
        c_rand = np.random.uniform(-0.9, 0.9, (N, 2))
        norms = np.linalg.norm(c_rand, axis=1)
        mask = norms > 0.9
        c_rand[mask] /= norms[mask, np.newaxis] * 0.9
        initial_configs.append(c_rand)

    for c_init in initial_configs:
        # Binary search for max r
        r_low = 0.0
        r_high = 0.2 # Safe upper bound
        
        # First, try to find a feasible r
        # Start with a small r and increase
        r = 0.05
        c_current = c_init.copy()
        
        # Relax to find a good configuration for small r
        c_current = relax(c_current, r)
        
        # Increase r gradually
        for _ in range(50):
            r_new = r + 0.005
            # Try to relax to r_new
            c_test = relax(c_current, r_new, max_iters=100)
            e_test, _ = compute_forces_and_energy(c_test, r_new)
            
            if e_test < 1e-6:
                r = r_new
                c_current = c_test
            else:
                # If failed, try smaller increment
                r_new = r + 0.001
                c_test = relax(c_current, r_new, max_iters=100)
                e_test, _ = compute_forces_and_energy(c_test, r_new)
                if e_test < 1e-6:
                    r = r_new
                    c_current = c_test
                else:
                    break
                    
        # Final polish
        if r > best_r:
            c_final = relax(c_current, r, max_iters=200)
            e_final, _ = compute_forces_and_energy(c_final, r)
            if e_final < 1e-6:
                best_r = r
                best_centers = c_final

    if best_centers is None:
        best_centers = make_initial_hexagonal(N, 0.15)
        
    return best_centers
# EVOLVE-BLOCK-END
