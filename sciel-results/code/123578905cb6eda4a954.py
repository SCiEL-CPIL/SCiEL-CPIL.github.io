import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf

        def count_call():
            nonlocal calls_made
            calls_made += 1

        def evaluate(x):
            nonlocal best_x, best_f
            x = np.clip(x, -5, 5)
            f = float(func(x))
            count_call()
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Differential Evolution for Global Search
        # NP = 20, Generations = 10 -> 200 calls
        NP = 20
        max_gen = 10
        F = 0.8
        CR = 0.9
        
        # Initialize population
        pop = rng.uniform(-5, 5, (NP, dim))
        pop_f = np.array([evaluate(ind) for ind in pop])
        
        if time.time() > deadline or calls_made >= budget:
            return best_x, best_f

        for g in range(max_gen):
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Sort population
            sort_idx = np.argsort(pop_f)
            pop = pop[sort_idx]
            pop_f = pop_f[sort_idx]
            
            for i in range(NP):
                if time.time() > deadline or calls_made >= budget:
                    break
                
                # Select three distinct random indices different from i
                r1, r2, r3 = rng.choice(NP, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(NP, 3, replace=False)
                
                # Create trial vector
                v = pop[r1] + F * (pop[r2] - pop[r3])
                trial = np.zeros(dim)
                for j in range(dim):
                    if rng.random() < CR:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                trial = np.clip(trial, -5, 5)
                trial_f = evaluate(trial)
                
                # Selection
                if trial_f <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f

        # Phase 2: Nelder-Mead Local Refinement
        # Take best 5 points from DE
        sort_idx = np.argsort(pop_f)
        best_starts = pop[sort_idx[:5]]
        best_starts_f = pop_f[sort_idx[:5]]
        
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f
            
        nm_budget_per_start = max(10, remaining_budget // 5)
        
        for start_idx in range(5):
            if time.time() > deadline or calls_made >= budget:
                break
            
            x0 = best_starts[start_idx].copy()
            f0 = best_starts_f[start_idx]
            
            # Initialize Simplex
            delta = 0.5
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += delta
                simplex.append(xi)
            
            simplex_f = [f0]
            for i in range(dim):
                if time.time() > deadline or calls_made >= budget:
                    break
                f_new = evaluate(simplex[i+1])
                simplex_f.append(f_new)
            
            if len(simplex_f) < dim + 1:
                continue
                
            # Nelder-Mead Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            max_iter = nm_budget_per_start
            
            for _ in range(max_iter):
                if time.time() > deadline or calls_made >= budget:
                    break
                
                # Sort
                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]
                
                # Check convergence
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-8:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = evaluate(xr)
                
                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    # Reflection accepted
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    if fr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + rho * (centroid - simplex[-1])
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    fc = evaluate(xc)
                    
                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if time.time() > deadline or calls_made >= budget:
                                break
                            simplex_f[i] = evaluate(simplex[i])
                        
                        # If we broke out of shrink loop early, break outer loop
                        if time.time() > deadline or calls_made >= budget:
                            break

        return best_x, best_f
