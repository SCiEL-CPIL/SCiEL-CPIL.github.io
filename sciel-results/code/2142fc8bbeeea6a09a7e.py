import numpy as np
import time
import concurrent.futures
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

def construct(instance):
    """
    Find low-energy structures for Dzugutov clusters using Repulsive Annealing.
    This method uses a repulsive potential to help the system escape local minima,
    gradually reducing the repulsion to settle into the global minimum.
    """
    n = instance["n"]
    potential = instance["potential"]
    
    # Time budget: 300s total / 6 instances = 50s per instance
    time_limit_per_instance = 45.0
    start_time = time.time()
    
    def generate_random_initial(n, seed):
        """Generate a random initial cluster with minimum distance constraint."""
        rng = np.random.default_rng(seed)
        radius = 2.0 * n ** (1.0 / 3.0)
        points = []
        # Center atom
        points.append(np.zeros(3))
        
        attempts = 0
        while len(points) < n and attempts < 10000:
            attempts += 1
            # Random point in sphere
            r = radius * rng.uniform(0.5, 1.0)
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(-1, 1)
            x = r * np.sqrt(1 - z**2) * np.cos(theta)
            y = r * np.sqrt(1 - z**2) * np.sin(theta)
            p = np.array([x, y, z])
            
            # Check distance
            if all(np.linalg.norm(p - q) > 1.0 for q in points):
                points.append(p)
                
        # If not enough points, add random ones (relaxation will fix)
        while len(points) < n:
            points.append(rng.normal(0, radius, 3))
            
        points = np.array(points[:n])
        points = points - np.mean(points, axis=0)
        return points

    def dzugutov_energy_repulsive(x, lam):
        """
        Compute Dzugutov energy + repulsive term.
        Repulsive term: sum 1/r^2 for all pairs.
        """
        pos = x.reshape(-1, 3)
        e_dz, grad_dz = dzugutov_energy_gradient(pos, potential)
        
        # Compute repulsive energy and gradient
        # V_rep = 1/r^2, dV_rep/dr = -2/r^3
        # F_rep = -dV/dr = 2/r^3 * r_hat
        # Note: We want to minimize, so we add to energy.
        
        n_atoms = len(pos)
        e_rep = 0.0
        grad_rep = np.zeros_like(pos)
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                rij = pos[i] - pos[j]
                r = np.linalg.norm(rij)
                if r < 1e-6:
                    # Avoid division by zero, add small repulsive force
                    r = 1e-6
                    rij = np.array([1e-6, 0, 0])
                
                # Only add repulsion if close? Or always?
                # Always is safer to prevent collapse.
                inv_r2 = 1.0 / (r*r)
                e_rep += inv_r2
                
                # Gradient of 1/r^2 wrt pos[i]
                # d(1/r^2)/d(rij) = -2 * rij / r^4
                # grad_i = -2 * rij / r^4
                # grad_j = 2 * rij / r^4
                factor = -2.0 * (1.0 / (r**4))
                grad_rep[i] += factor * rij
                grad_rep[j] -= factor * rij
                
        e_total = e_dz + lam * e_rep
        grad_total = grad_dz.flatten() + lam * grad_rep.flatten()
        
        return e_total, grad_total

    def local_minimize_repulsive(x, lam):
        """Minimize E_dz + lam * E_rep."""
        res = minimize(
            fun=dzugutov_energy_repulsive,
            x0=x,
            args=(lam,),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8, 'maxls': 20}
        )
        return res.fun, res.x.reshape(-1, 3)

    def repulsive_annealing_trajectory(seed):
        """Run a single Repulsive Annealing trajectory."""
        rng = np.random.default_rng(seed)
        
        # Initial configuration
        x = generate_random_initial(n, seed)
        
        # Annealing parameters
        lam_start = 5.0
        lam_end = 0.0
        n_steps = 40
        # Geometric cooling
        if n_steps > 1:
            ratio = (lam_end / lam_start) ** (1.0 / n_steps) if lam_end > 0 else 0.0
            # If lam_end is 0, we can't use geometric ratio directly.
            # Use linear or exponential decay to 0.
            # Let's use exponential decay: lam_k = lam_start * exp(-k * alpha)
            # Or simply: lam_k = lam_start * (1 - k/n_steps)^2
            pass
            
        # Schedule: lam_k = lam_start * (1 - k/n_steps)^2
        best_e = float('inf')
        best_x = x.copy()
        
        for k in range(n_steps):
            if time.time() - start_time > time_limit_per_instance - 2:
                break
                
            # Calculate current lambda
            progress = k / n_steps
            lam = lam_start * (1.0 - progress) ** 2
            
            # Minimize
            try:
                e_curr, x_curr = local_minimize_repulsive(x, lam)
            except Exception:
                # If minimization fails, try a smaller step or break
                break
                
            # Update best
            if e_curr < best_e:
                best_e = e_curr
                best_x = x_curr.copy()
                
            x = x_curr
            
        # Final relaxation without repulsion
        try:
            e_final, x_final = local_minimize_repulsive(x, 0.0)
            if e_final < best_e:
                best_e = e_final
                best_x = x_final
        except Exception:
            pass
            
        return best_e, best_x

    n_threads = 4
    best_energy = float('inf')
    best_positions = None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for i in range(n_threads):
            seed = 100 + i * 100
            futures.append(executor.submit(repulsive_annealing_trajectory, seed))
            
        for future in concurrent.futures.as_completed(futures):
            try:
                e, pos = future.result()
                if e < best_energy:
                    best_energy = e
                    best_positions = pos
            except Exception:
                pass
    
    if best_positions is None:
        # Fallback
        rng = np.random.default_rng(n)
        points = rng.normal(0, 1, (n, 3))
        points = points - np.mean(points, axis=0)
        _, best_positions = relax(points, potential)
        
    return best_positions

# EVOLVE-BLOCK-END
