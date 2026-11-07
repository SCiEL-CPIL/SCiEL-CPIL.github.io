import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def eval_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or not time_ok():
                return np.inf
            calls += 1
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Differential Evolution
        # Use ~80% of budget for DE
        de_budget = int(budget * 0.8)
        
        NP = 12
        F_start = 0.9
        F_end = 0.3
        CR = 0.9
        
        population = np.zeros((NP, dim))
        fitness = np.zeros(NP)
        
        for i in range(NP):
            x = rng.uniform(-4.0, 4.0, size=dim)
            population[i] = x
            fitness[i] = eval_func(x)
            
        idx_best = np.argmin(fitness)
        current_best_f = fitness[idx_best]
        current_best_x = population[idx_best].copy()
        
        total_de_generations = (de_budget - NP) // NP
        gen = 0
        
        while calls < de_budget and time_ok() and gen < total_de_generations:
            # Adaptive F
            F = F_start + (F_end - F_start) * (gen / max(1, total_de_generations))
            
            improved = False
            for i in range(NP):
                if calls >= de_budget or not time_ok():
                    break
                
                candidates = [j for j in range(NP) if j != i]
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                
                v = population[r1] + F * (population[r2] - population[r3])
                v = np.clip(v, -5.0, 5.0)
                
                trial = population[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                        
                f_trial = eval_func(trial)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < current_best_f:
                        current_best_f = f_trial
                        current_best_x = trial.copy()
                        improved = True
            
            # Stagnation handling: if no improvement in this generation, perturb worst
            if not improved and gen > 2:
                worst_idx = np.argmax(fitness)
                # Small random perturbation around best
                population[worst_idx] = current_best_x + rng.normal(0, 0.1, size=dim)
                population[worst_idx] = np.clip(population[worst_idx], -5.0, 5.0)
                fitness[worst_idx] = eval_func(population[worst_idx])
                if fitness[worst_idx] < current_best_f:
                    current_best_f = fitness[worst_idx]
                    current_best_x = population[worst_idx].copy()

            gen += 1

        # Phase 2: Local Polish with Nelder-Mead
        # Use remaining budget
        if calls < budget and time_ok():
            x_start = current_best_x.copy()
            f_start = current_best_f
            
            # Initialize simplex
            simplex = [x_start.copy()]
            simplex_f = [f_start]
            
            # Add vertices
            for i in range(dim):
                x_new = x_start.copy()
                # Step size based on initial variance or fixed
                step = 0.5 if abs(x_start[i]) < 0.5 else 0.1
                x_new[i] += step
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = eval_func(x_new)
                simplex.append(x_new)
                simplex_f.append(f_new)
                
                if f_new < f_start:
                    f_start = f_new
                    x_start = x_new.copy()
                    # Update simplex base? No, just let NM handle it.
                    # But we should keep the best point in the simplex.
                    # Actually, standard NM implementation below handles this.
            
            # Re-sort simplex by fitness
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Nelder-Mead Loop
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            while calls < budget and time_ok():
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = eval_func(x_ref)
                
                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = eval_func(x_exp)
                    
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
                    if f_ref < simplex_f[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (x_ref - centroid)
                    
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = eval_func(x_con)
                    
                    if f_con < min(f_ref, simplex_f[-1]):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            simplex_f[i] = eval_func(simplex[i])
                
                # Re-sort
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Update global best
                if simplex_f[0] < current_best_f:
                    current_best_f = simplex_f[0]
                    current_best_x = simplex[0].copy()
                    
                # Check for convergence (optional, but budget limit is hard)
                # Simple check: if simplex volume is small
                # For now, just run until budget or time out

        return best_x, best_f
