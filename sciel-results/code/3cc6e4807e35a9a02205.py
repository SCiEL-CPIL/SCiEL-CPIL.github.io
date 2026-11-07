import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        try:
            # Phase 1: Differential Evolution for global exploration
            pop_size = 20
            F = 0.8  # Differential weight
            CR = 0.9  # Crossover rate
            pop = [rng.uniform(-5, 5, dim) for _ in range(pop_size)]
            pop_f = np.array([func(x) for x in pop])
            calls_made += pop_size
            best_idx = np.argmin(pop_f)
            best_f = pop_f[best_idx]
            best_x = pop[best_idx].copy()
            
            # Track best per individual for DE
            pop_best = [x.copy() for x in pop]
            pop_best_f = pop_f.copy()
            
            # Phase 1 budget: use ~60% of total budget for DE
            de_budget = int(budget * 0.6)
            de_generations = 0
            no_improve_gen = 0
            stagnation_threshold = 30
            
            while calls_made < de_budget and time.time() < deadline:
                # Generate new population
                new_pop = []
                new_pop_f = []
                for i in range(pop_size):
                    if calls_made >= de_budget or time.time() >= deadline:
                        break
                    # Select three distinct random individuals
                    while True:
                        r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                        if len(set([r1, r2, r3])) == 3 and i not in [r1, r2, r3]:
                            break
                    
                    # Differential mutation
                    v = pop[r1] + F * (pop[r2] - pop[r3])
                    v = np.clip(v, -5, 5)
                    
                    # Binary crossover
                    trial = pop[i].copy()
                    j_rand = rng.integers(dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                    trial = np.clip(trial, -5, 5)
                    
                    # Evaluate
                    f_trial = func(trial)
                    calls_made += 1
                    
                    # Selection
                    if f_trial <= pop_f[i]:
                        new_pop.append(trial)
                        new_pop_f.append(f_trial)
                        pop_best[i] = trial.copy()
                        pop_best_f[i] = f_trial
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = trial.copy()
                    else:
                        new_pop.append(pop[i].copy())
                        new_pop_f.append(pop_f[i])
                
                if len(new_pop) < pop_size:
                    break
                    
                pop = new_pop
                pop_f = np.array(new_pop_f)
                de_generations += 1
                
                # Check for stagnation
                current_best = np.min(pop_f)
                if current_best < best_f - 1e-10:
                    no_improve_gen = 0
                else:
                    no_improve_gen += 1
                    
                # Restart DE if stagnated
                if no_improve_gen > stagnation_threshold:
                    # Reset population around best point with some diversity
                    pop = []
                    pop_f = []
                    for i in range(pop_size):
                        if i == 0:
                            x = best_x.copy()
                        else:
                            x = best_x + rng.normal(0, 2.0, dim)
                            x = np.clip(x, -5, 5)
                        fx = func(x)
                        calls_made += 1
                        pop.append(x)
                        pop_f.append(fx)
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    pop = [x.copy() for x in pop]
                    pop_f = np.array(pop_f)
                    pop_best = [x.copy() for x in pop]
                    pop_best_f = pop_f.copy()
                    no_improve_gen = 0

            # Phase 2: CMA-ES for local refinement
            # Use remaining budget
            cma_budget = budget - calls_made
            if cma_budget < 10:
                cma_budget = 10
            
            # Initialize CMA-ES around the best point
            lam = 10
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)

            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            dampsigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma

            mean = best_x.copy()
            C = np.eye(dim)
            C_sqrt = np.linalg.cholesky(C)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            sigma = 0.5  # Start with smaller step size for refinement
            
            cma_gen = 0
            cma_no_improve = 0
            
            while calls_made < budget and time.time() < deadline:
                if calls_made + lam > budget:
                    break
                    
                # Generate population
                offspring = []
                for i in range(lam):
                    if calls_made >= budget or time.time() >= deadline:
                        break
                    z = rng.standard_normal(dim)
                    x_off = mean + sigma * (C_sqrt @ z)
                    x_off = np.clip(x_off, -5, 5)
                    fx_off = func(x_off)
                    calls_made += 1
                    offspring.append((x_off, fx_off))
                
                if len(offspring) < mu:
                    break
                
                # Sort offspring
                offspring.sort(key=lambda x: x[1])
                best_offspring_x, best_offspring_f = offspring[0]
                
                # Update global best
                if best_offspring_f < best_f:
                    best_f = best_offspring_f
                    best_x = best_offspring_x.copy()
                    cma_no_improve = 0
                else:
                    cma_no_improve += 1
                    
                # Check for stagnation in CMA-ES
                if cma_no_improve > 20:
                    # Restart CMA-ES with smaller sigma
                    sigma = max(sigma * 0.5, 0.01)
                    cma_no_improve = 0
                    # Reset covariance if sigma is very small
                    if sigma < 0.05:
                        C = np.eye(dim)
                        C_sqrt = np.linalg.cholesky(C)
                        p_sigma = np.zeros(dim)
                        p_c = np.zeros(dim)
                    
                # Standard CMA-ES update
                x_w = np.zeros(dim)
                for i in range(mu):
                    x_w += weights[i] * offspring[i][0]
                
                # Update evolution paths
                try:
                    C_sqrt_inv = np.linalg.inv(C_sqrt)
                except:
                    break
                
                y_w = (x_w - mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_sqrt_inv @ y_w)
                
                hsig = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (cma_gen + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim) else 0.0
                p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * (C_sqrt_inv @ y_w)
                
                # Update mean
                mean = x_w
                
                # Update covariance
                rank_one = np.outer(p_sigma, p_sigma)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = C_sqrt_inv @ (offspring[i][0] - mean) / sigma
                    rank_mu += weights[i] * np.outer(y_i, y_i)
                
                C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                C = (C + C.T) / 2
                
                # Ensure PSD
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Update sigma
                sigma = sigma * np.exp(c_sigma / dampsigma * (np.linalg.norm(p_sigma) / np.sqrt(dim) - 1))
                
                # Update Cholesky
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except:
                    C = np.eye(dim)
                    C_sqrt = np.linalg.cholesky(C)
                    sigma = 0.1
                
                cma_gen += 1

        except Exception:
            pass

        return best_x, best_f
