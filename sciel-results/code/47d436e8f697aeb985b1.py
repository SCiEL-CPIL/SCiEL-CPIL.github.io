"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-restart CMA-ES with Nelder-Mead and DE phases for f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def check_time():
            return (time.time() - start_time) < time_limit

        # Global best tracking
        x_global_best = None
        f_global_best = np.inf
        calls = 0

        # Helper to evaluate and update global best
        def evaluate(x):
            nonlocal calls, f_global_best, x_global_best
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            if f < f_global_best:
                f_global_best = f
                x_global_best = x.copy()
            return f

        # Phase 1: Initial random sampling to find a promising region
        n_init = max(10, budget // 10)
        
        x0 = rng.uniform(-5, 5, dim)
        evaluate(x0)
        
        for _ in range(n_init - 1):
            if not check_time() or calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        # Phase 2: Multi-restart CMA-ES (1+1 with covariance)
        max_restarts = 5
        restart_budget = max(20, budget // (max_restarts + 1))
        
        for restart in range(max_restarts):
            if not check_time() or calls >= budget:
                break
            
            x = x_global_best.copy()
            f = f_global_best
            sigma = 2.0
            C = np.eye(dim)
            local_calls = 0
            local_limit = min(restart_budget, budget - calls)
            stall_count = 0
            stall_threshold = 10
            
            while local_calls < local_limit and check_time() and calls < budget:
                z = rng.multivariate_normal(np.zeros(dim), C)
                norm_z = np.linalg.norm(z)
                if norm_z > 1e-10:
                    z = z / norm_z
                
                x_trial = x + sigma * z
                f_trial = evaluate(x_trial)
                local_calls += 1
                
                if f_trial < f:
                    C = (1 - 1.0/dim) * C + (1.0/dim) * (np.outer(z, z))
                    x = x_trial
                    f = f_trial
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count >= stall_threshold:
                        sigma *= 0.7
                        stall_count = 0
                        C = 0.5 * C + 0.5 * np.eye(dim)

        # Phase 3: Nelder-Mead local refinement on global best
        if check_time() and calls < budget:
            try:
                # Use Nelder-Mead to refine the best point found so far
                result = minimize(
                    func,
                    x_global_best,
                    method='Nelder-Mead',
                    options={'maxiter': min(50, budget - calls), 'xatol': 1e-6, 'fatol': 1e-8}
                )
                # Count the calls made by Nelder-Mead
                # scipy's minimize doesn't easily expose call count, so we estimate
                # We'll just evaluate the result to see if it's better
                x_nm = np.clip(result.x, -5.0, 5.0)
                f_nm = func(x_nm)
                calls += 1
                if f_nm < f_global_best:
                    f_global_best = f_nm
                    x_global_best = x_nm.copy()
            except Exception:
                pass

        # Phase 4: Differential Evolution for alternative basins
        if check_time() and calls < budget:
            try:
                # Simple DE implementation
                pop_size = 10
                F = 0.8  # scale factor
                CR = 0.9  # crossover rate
                
                # Initialize population around global best and random points
                pop = [x_global_best.copy()]
                for _ in range(pop_size - 1):
                    if rng.random() < 0.5:
                        # Perturb global best
                        x = x_global_best + rng.normal(0, 1.0, dim)
                    else:
                        # Random point
                        x = rng.uniform(-5, 5, dim)
                    x = np.clip(x, -5.0, 5.0)
                    pop.append(x)
                
                # Evaluate initial population
                pop_f = [evaluate(x) for x in pop]
                
                de_generations = 20
                for gen in range(de_generations):
                    if not check_time() or calls >= budget:
                        break
                    
                    # Find best in population
                    best_idx = np.argmin(pop_f)
                    best_x = pop[best_idx].copy()
                    best_f = pop_f[best_idx]
                    
                    # Update global best
                    if best_f < f_global_best:
                        f_global_best = best_f
                        x_global_best = best_x.copy()
                    
                    # Generate new population
                    new_pop = [best_x.copy()]
                    new_pop_f = [best_f]
                    
                    for i in range(pop_size - 1):
                        # Select 3 distinct random individuals
                        a, b, c = rng.choice(pop_size, 3, replace=False)
                        
                        # Mutation
                        v = pop[a] + F * (pop[b] - pop[c])
                        
                        # Crossover
                        trial = rng.random(dim) < CR
                        j_rand = rng.integers(dim)
                        trial[j_rand] = True
                        trial_x = np.where(trial, v, pop[i])
                        trial_x = np.clip(trial_x, -5.0, 5.0)
                        
                        # Selection
                        f_trial = evaluate(trial_x)
                        if f_trial <= pop_f[i]:
                            new_pop.append(trial_x)
                            new_pop_f.append(f_trial)
                        else:
                            new_pop.append(pop[i].copy())
                            new_pop_f.append(pop_f[i])
                    
                    pop = new_pop
                    pop_f = new_pop_f
            
            except Exception:
                pass

        return x_global_best, f_global_best
# EVOLVE-BLOCK-END
