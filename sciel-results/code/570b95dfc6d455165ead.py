"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start DE + Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        dim = self.dim
        budget = self.budget
        
        # Count calls to ensure we don't exceed budget
        calls_made = 0
        
        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5, 5)
            calls_made += 1
            try:
                return float(func(x))
            except Exception:
                return np.inf

        # Differential Evolution parameters
        # Since dim=5 is small, population size can be modest
        pop_size = 10
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        
        # Initialize population
        rng = np.random.default_rng(42)
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fit = np.array([safe_call(x) for x in pop])
        
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]
        
        # DE Loop
        while calls_made < budget - 20:  # Reserve some calls for refinement
            if time.time() > deadline:
                break
                
            for i in range(pop_size):
                if calls_made >= budget - 20:
                    break
                if time.time() > deadline:
                    break
                    
                # Select three distinct random individuals
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                    
                # Generate trial vector
                v = pop[r1] + F * (pop[r2] - pop[r3])
                v = np.clip(v, -5, 5)
                
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Evaluation
                trial_f = safe_call(trial)
                
                # Selection
                if trial_f <= fit[i]:
                    pop[i] = trial
                    fit[i] = trial_f
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial.copy()
        
        # Nelder-Mead refinement on the best point found
        if calls_made < budget - 10 and time.time() < deadline:
            # Initialize simplex
            simplex = [best_x.copy() for _ in range(dim + 1)]
            simplex_f = [best_f for _ in range(dim + 1)]
            
            for i in range(1, dim + 1):
                simplex[i][i-1] += 0.1
                simplex[i] = np.clip(simplex[i], -5, 5)
                simplex_f[i] = safe_call(simplex[i])
                
                if simplex_f[i] < best_f:
                    best_f = simplex_f[i]
                    best_x = simplex[i].copy()
            
            # NM Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            max_iter = 20
            for _ in range(max_iter):
                if calls_made >= budget - 5 or time.time() > deadline:
                    break
                    
                # Sort simplex
                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = safe_call(x_ref)
                
                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = safe_call(x_exp)
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + 0.5 * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = safe_call(x_con)
                    
                    if f_con < simplex_f[-1]:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = safe_call(simplex[i])
                
                # Update best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_f:
                    best_f = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
