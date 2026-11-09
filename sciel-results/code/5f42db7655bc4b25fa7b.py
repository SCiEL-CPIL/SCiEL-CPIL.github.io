"""Island Model CMA-ES for Katsuura f23 with migration."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """
    Optimizes Katsuura f23 using an island model of CMA-ES instances.
    1. Initialize multiple CMA-ES islands with different starting points.
    2. Run islands in parallel (sequentially) with periodic migration of best solutions.
    3. This structural change aims to escape local minima via diversity maintenance.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        lower = -5.0
        upper = 5.0
        
        calls_made = 0
        x_global_best = None
        f_global_best = np.inf
        
        def check_deadline():
            return time.time() > deadline
            
        def evaluate(x):
            nonlocal calls_made, x_global_best, f_global_best
            x_clipped = np.clip(x, lower, upper)
            fx = func(x_clipped)
            calls_made += 1
            if fx < f_global_best:
                f_global_best = fx
                x_global_best = x_clipped.copy()
            return x_clipped, fx

        # --- Initialize Islands ---
        n_islands = 4
        budget_per_island = max(1, (budget - n_islands) // n_islands)
        
        islands = []
        for i in range(n_islands):
            if calls_made >= budget or check_deadline():
                break
                
            # Start each island at a different random point
            x0 = rng.uniform(lower, upper, size=dim)
            x0, f0 = evaluate(x0)
            
            # Initialize CMA-ES state for this island
            island = {
                'x_mean': x0.copy(),
                'sigma': 1.0,
                'C': np.eye(dim),
                'p_sigma': np.zeros(dim),
                'p_c': np.zeros(dim),
                'best_x': x0.copy(),
                'best_f': f0,
                'calls': 1,
                'no_improve': 0
            }
            islands.append(island)
        
        if len(islands) == 0:
            return x_global_best, f_global_best
            
        # Migration parameters
        migration_interval = 10  # Generations between migrations
        migration_rate = 1  # Number of migrants
        
        total_generations = 0
        max_generations = 500  # Safety cap
        
        while calls_made < budget and not check_deadline() and total_generations < max_generations:
            total_generations += 1
            any_progress = False
            
            # Run one generation on each island
            for idx, island in enumerate(islands):
                if calls_made >= budget or check_deadline():
                    break
                    
                dim = self.dim
                x_mean = island['x_mean']
                sigma = island['sigma']
                C = island['C']
                p_sigma = island['p_sigma']
                p_c = island['p_c']
                
                # Fixed population size for each island
                lam = 4
                mu = 2
                
                # Recalculate weights
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w = w / np.sum(w)
                mu_eff = 1.0 / np.sum(w**2)
                
                c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
                d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
                c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
                c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
                c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
                
                # Generate population
                ys = []
                fs = []
                for i in range(lam):
                    if calls_made >= budget or check_deadline():
                        break
                        
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = np.eye(dim)
                        L = np.eye(dim)
                    
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (L @ z)
                    y, fy = evaluate(y)
                    ys.append(y)
                    fs.append(fy)
                
                if len(ys) == 0:
                    break
                    
                island['calls'] += len(ys)
                
                # Sort by function value
                indices = np.argsort(fs)
                ys_sorted = [ys[i] for i in indices]
                fs_sorted = [fs[i] for i in indices]
                
                # Update island best
                if fs_sorted[0] < island['best_f']:
                    island['best_f'] = fs_sorted[0]
                    island['best_x'] = ys_sorted[0].copy()
                    island['no_improve'] = 0
                    any_progress = True
                else:
                    island['no_improve'] += 1
                    
                # Update mean
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += w[i] * ys_sorted[i]
                x_mean = y_w
                
                # Update sigma control
                y_sigma = (L.T @ (y_w / sigma))
                p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                
                # Hessian potential
                h_sigma = 1.0 if (np.linalg.norm(p_sigma) / math.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * math.sqrt(mu_eff + 1 - 1)) else 0.0
                
                sigma *= math.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / math.sqrt(mu_eff) - 1))
                
                # Update covariance control
                y_c = (L.T @ (y_w / sigma))
                p_c = (1 - c_c) * p_c + math.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
                
                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = (1 - c_1 - c_mu + delta_h) * C
                C += c_1 * (np.outer(p_c, p_c) + delta_h * np.eye(dim))
                for i in range(mu):
                    y_i = (L.T @ (ys_sorted[i] / sigma))
                    C += c_mu * w[i] * np.outer(y_i, y_i)
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    if np.min(eigvals) < 1e-10:
                        C = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                
                # Update island state
                island['x_mean'] = x_mean
                island['sigma'] = sigma
                island['C'] = C
                island['p_sigma'] = p_sigma
                island['p_c'] = p_c
                
                # Reset island if stuck for too long
                if island['no_improve'] > 30:
                    # Reset to global best or random point
                    if x_global_best is not None:
                        x_new = x_global_best + rng.normal(0, 0.5, size=dim)
                    else:
                        x_new = rng.uniform(lower, upper, size=dim)
                    x_new = np.clip(x_new, lower, upper)
                    _, f_new = evaluate(x_new)
                    
                    island['x_mean'] = x_new.copy()
                    island['sigma'] = 1.0
                    island['C'] = np.eye(dim)
                    island['p_sigma'] = np.zeros(dim)
                    island['p_c'] = np.zeros(dim)
                    island['best_x'] = x_new.copy()
                    island['best_f'] = f_new
                    island['no_improve'] = 0
                    island['calls'] = 1
            
            # Migration
            if total_generations % migration_interval == 0 and len(islands) > 1:
                # Sort islands by best fitness
                sorted_islands = sorted(range(len(islands)), key=lambda i: islands[i]['best_f'])
                
                # Migrate best from top island to worst island
                if len(sorted_islands) >= 2:
                    best_idx = sorted_islands[0]
                    worst_idx = sorted_islands[-1]
                    
                    # Replace worst island's best with best island's best (with perturbation)
                    if islands[worst_idx]['best_f'] > islands[best_idx]['best_f']:
                        x_migrate = islands[best_idx]['best_x'] + rng.normal(0, 0.2, size=dim)
                        x_migrate = np.clip(x_migrate, lower, upper)
                        _, f_migrate = evaluate(x_migrate)
                        
                        # Update worst island with migrant if better
                        if f_migrate < islands[worst_idx]['best_f']:
                            islands[worst_idx]['best_x'] = x_migrate.copy()
                            islands[worst_idx]['best_f'] = f_migrate
                            # Also update mean to start from migrant
                            islands[worst_idx]['x_mean'] = x_migrate.copy()
                            islands[worst_idx]['sigma'] = 1.0
                            islands[worst_idx]['C'] = np.eye(dim)
                            islands[worst_idx]['p_sigma'] = np.zeros(dim)
                            islands[worst_idx]['p_c'] = np.zeros(dim)
                            islands[worst_idx]['no_improve'] = 0

        return x_global_best, f_global_best
# EVOLVE-BLOCK-END
