import numpy as np
import time
import concurrent.futures
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

def construct(instance):
    """
    Find low-energy structures for Dzugutov clusters using Basin Hopping.
    Basin Hopping: kick atoms randomly, local minimize, Metropolis accept/reject.
    """
    n = instance["n"]
    potential = instance["potential"]
    
    # Time budget: 300s total / 6 instances = 50s per instance
    time_limit_per_instance = 45.0
    start_time = time.time()
    
    def local_minimize(x):
        """Local minimize using L-BFGS-B with Dzugutov potential."""
        def obj_and_grad(flat_x):
            pos = flat_x.reshape(-1, 3)
            e, g = dzugutov_energy_gradient(pos, potential)
            return e, g.flatten()
        
        try:
            res = minimize(
                fun=obj_and_grad,
                x0=x.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20}
            )
            return res.fun, res.x.reshape(-1, 3)
        except Exception:
            return float('inf'), x
    
    def basin_hopping_trajectory(seed, time_limit):
        """Run a single Basin Hopping trajectory."""
        rng = np.random.default_rng(seed)
        
        # Initial configuration: random in sphere
        radius = 1.5 * n ** (1.0 / 3.0)
        x = rng.uniform(-radius, radius, (n, 3))
        x = x - np.mean(x, axis=0)
        
        # Local minimize
        e_curr, x_curr = local_minimize(x)
        
        best_e = e_curr
        best_x = x_curr.copy()
        
        # Basin hopping parameters
        kick_start = 0.5
        kick_end = 0.05
        T_start = 0.5
        T_end = 0.01
        n_steps = 300
        
        for k in range(n_steps):
            if time.time() - start_time > time_limit:
                break
            
            # Anneal kick size and temperature
            progress = k / max(n_steps, 1)
            kick_size = kick_start + (kick_end - kick_start) * progress
            T = T_start + (T_end - T_start) * progress
            
            # Kick
            x_kicked = x_curr + rng.normal(0, kick_size, (n, 3))
            
            # Local minimize
            e_new, x_new = local_minimize(x_kicked)
            
            # Metropolis acceptance
            if e_new < e_curr:
                x_curr = x_new
                e_curr = e_new
                if e_new < best_e:
                    best_e = e_new
                    best_x = x_new.copy()
            else:
                delta_e = e_new - e_curr
                if T > 1e-10 and delta_e / T < 50:
                    prob = np.exp(-delta_e / T)
                    if rng.random() < prob:
                        x_curr = x_new
                        e_curr = e_new
                        if e_new < best_e:
                            best_e = e_new
                            best_x = x_new.copy()
        
        return best_e, best_x
    
    n_threads = 4
    best_energy = float('inf')
    best_positions = None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for i in range(n_threads):
            seed = 42 + i * 1000
            futures.append(executor.submit(basin_hopping_trajectory, seed, time_limit_per_instance))
        
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
