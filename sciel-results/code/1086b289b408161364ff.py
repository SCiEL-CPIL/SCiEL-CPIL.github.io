"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Population-based DE with adaptive mutation and Nelder-Mead polish.
    Strategy:
    1. Initialize a diverse population across [-5, 5]^dim.
    2. Run DE/rand/1 with adaptive mutation factor and binomial crossover.
    3. Polish the global best with Nelder-Mead.
    4. Use remaining budget for (1+1)-ES refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        dim = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            return float(func(x))

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: DE/rand/1 with adaptive mutation
        # Population size: balance exploration vs exploitation
        pop_size = 12
        # Use remaining budget for DE generations
        # Each generation costs ~2*pop_size evaluations (mutant + trial)
        # Reserve ~50 for polish
        de_budget = max(50, budget - 50)
        n_gen = max(1, de_budget // (2 * pop_size))
        
        # Initialize population: mix of center, random, and structured
        pop = []
        # Center point
        pop.append(np.zeros(dim))
        # Random points
        for _ in range(pop_size - 1):
            pop.append(rng.uniform(-5, 5, dim))
        
        # Evaluate initial population
        pop_f = []
        for p in pop:
            if calls[0] >= budget or time.time() > deadline:
                break
            try:
                f_val = wrapped(p)
                pop_f.append(f_val)
                update_best(p, f_val)
            except Exception:
                break
        
        if len(pop_f) < pop_size:
            pop = pop[:len(pop_f)]
            pop_size = len(pop)
        
        if pop_size < 3:
            # Not enough points for DE, just do (1+1)-ES
            if best_x is not None and calls[0] < budget:
                x = best_x.copy()
                fx = best_f
                sigma = 1.0
                no_improve = 0
                max_no_improve = 30
                
                while calls[0] < budget and time.time() < deadline:
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    try:
                        fy = wrapped(y)
                    except Exception:
                        break
                    
                    if fy < fx:
                        x, fx = y, fy
                        update_best(x, fx)
                        sigma *= 1.05
                        no_improve = 0
                    else:
                        sigma *= 0.95
                        no_improve += 1
                    
                    if no_improve > max_no_improve:
                        break
                    if sigma < 1e-6:
                        break

            if best_x is None:
                best_x = rng.uniform(-5, 5, dim)
                best_f = wrapped(best_x)

            return best_x, best_f

        # DE Loop with adaptive mutation factor
        F = 0.7  # Initial mutation factor
        CR = 0.9  # Crossover rate (high for better exploration)
        
        for gen in range(n_gen):
            if calls[0] >= budget or time.time() > deadline:
                break
            
            new_pop = []
            new_pop_f = []
            
            for i in range(pop_size):
                if calls[0] >= budget or time.time() > deadline:
                    break
                
                # Select three distinct random individuals
                indices = list(range(pop_size))
                rng.shuffle(indices)
                r1, r2, r3 = indices[0], indices[1], indices[2]
                
                # Adaptive mutation: adjust F based on progress
                # If best hasn't improved recently, increase F for more exploration
                # Simple heuristic: keep F in [0.5, 1.0]
                F_current = np.clip(F + 0.1 * (gen % 3 - 1), 0.5, 1.0)
                
                # Mutation: DE/rand/1
                v = pop[r1] + F_current * (pop[r2] - pop[r3])
                v = np.clip(v, -5, 5)
                
                # Evaluate mutant (optional, can skip to save budget)
                # For budget efficiency, skip mutant evaluation and go straight to trial
                # But let's evaluate to have a better base for crossover
                try:
                    f_v = wrapped(v)
                except Exception:
                    continue
                
                # Crossover: binomial
                u = np.zeros(dim)
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                u = np.clip(u, -5, 5)
                
                # Evaluate candidate
                try:
                    f_u = wrapped(u)
                except Exception:
                    continue
                
                # Selection
                if f_u <= pop_f[i]:
                    new_pop.append(u)
                    new_pop_f.append(f_u)
                    update_best(u, f_u)
                else:
                    new_pop.append(pop[i])
                    new_pop_f.append(pop_f[i])
            
            # If we broke out early due to budget, break the generation loop
            if len(new_pop) < pop_size:
                break
                
            pop = new_pop
            pop_f = new_pop_f
            
            # Update F: if we made progress, decrease F slightly for exploitation
            # If no progress, increase F for exploration
            # Simple: keep F constant or slowly adapt
            if gen % 5 == 4:
                # Check if best improved in last 5 generations
                # For simplicity, just oscillate F slightly
                F = np.clip(F + 0.05 * (rng.random() - 0.5), 0.5, 1.0)

        # Phase 2: Polish the best point with Nelder-Mead
        if best_x is not None and calls[0] < budget:
            try:
                remaining = budget - calls[0]
                nm_maxiter = min(200, max(20, remaining // 2))
                
                res_nm = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': nm_maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                update_best(res_nm.x, res_nm.fun)
            except Exception:
                pass

        # Phase 3: (1+1)-ES refinement from the best point
        if best_x is not None and calls[0] < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05  # Start with small step for polishing
            no_improve = 0
            max_no_improve = 30
            
            while calls[0] < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    fy = wrapped(y)
                except Exception:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.05
                    no_improve = 0
                else:
                    sigma *= 0.95
                    no_improve += 1
                
                if no_improve > max_no_improve:
                    break
                if sigma < 1e-8:
                    break

        # Phase 4: Final tight polish if budget allows
        if best_x is not None and calls[0] < budget - 10:
            try:
                res_final = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                update_best(res_final.x, res_final.fun)
            except Exception:
                pass

        if best_x is None:
            # Fallback
            best_x = rng.uniform(-5, 5, dim)
            best_f = wrapped(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
