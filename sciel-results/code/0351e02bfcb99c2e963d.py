"""
Multi-restart CMA-ES optimizer for BBOB f3 (Rastrigin).
Uses multiple independent CMA-ES runs with different initializations to
handle the multimodality of Rastrigin. Each restart gets a portion of
the budget, and the best result across all restarts is returned.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin

        n = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0
        rng = np.random.default_rng(42)

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= budget

        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def run_de(max_calls):
            """Run Differential Evolution for global exploration."""
            nonlocal best_x, best_f, calls_made
            
            pop_size = 20
            if pop_size > max_calls:
                pop_size = max(4, max_calls // 5)
            
            # Initialize population uniformly in [-5, 5]
            population = rng.uniform(-5, 5, size=(pop_size, n))
            fitnesses = np.empty(pop_size)
            
            for i in range(pop_size):
                fitnesses[i] = func(population[i])
                update_best(population[i], fitnesses[i])
                
            F = 0.8
            CR = 0.5
            
            while not check_deadline():
                # Check if we have enough budget for a full generation
                # If not, we can still run partial, but let's break if we can't afford at least 1 eval
                if calls_made >= budget:
                    break

                new_fitnesses = fitnesses.copy()
                new_population = population.copy()
                
                for i in range(pop_size):
                    if check_deadline():
                        break
                        
                    # Select 3 distinct random individuals
                    r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                    
                    # Mutation
                    v = population[r1] + F * (population[r2] - population[r3])
                    
                    # Crossover
                    u = rng.random(n) < CR
                    trial = v[u]
                    trial[~u] = population[i][~u]
                    
                    # Clip to bounds
                    trial = np.clip(trial, -5, 5)
                    
                    # Evaluate
                    f_trial = func(trial)
                    update_best(trial, f_trial)
                    
                    # Selection
                    if f_trial <= fitnesses[i]:
                        new_population[i] = trial
                        new_fitnesses[i] = f_trial
                        
                population = new_population
                fitnesses = new_fitnesses

        def run_cmaes(initial_mean, initial_sigma, max_calls):
            """Run CMA-ES for local refinement."""
            nonlocal best_x, best_f, calls_made
            
            pop_size = 4 + int(3 * np.log(n))
            if pop_size > max_calls:
                pop_size = max(2, max_calls // 2)
            
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)

            mean = initial_mean.copy()
            sigma = initial_sigma
            
            C = np.eye(n)
            C_sqrt = np.eye(n)
            
            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            c1 = 2 / ((n + 1.3)**2 + mu_eff)
            cmu = min(1 - c1, 2 * (weights[1] - 1) / ((n + 2)**2 + mu_eff))
            
            ps = np.zeros(n)
            pc = np.zeros(n)
            
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            no_improve = 0
            last_best = float('inf')
            
            while not check_deadline():
                # Generate offspring
                offspring = []
                for i in range(pop_size):
                    z = rng.standard_normal(n)
                    y = mean + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    update_best(y, fy)
                    offspring.append((fy, y))
                    
                    if check_deadline():
                        break
                
                if len(offspring) < mu:
                    break
                    
                # Sort offspring
                offspring.sort(key=lambda x: x[0])
                
                # Check improvement
                if offspring[0][0] < last_best - 1e-10:
                    last_best = offspring[0][0]
                    no_improve = 0
                else:
                    no_improve += 1
                
                # Recombination
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * offspring[i][1]
                    
                # Update mean
                old_mean = mean.copy()
                mean = y_w
                
                # Update covariance matrix
                ys = (y_w - old_mean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * ys
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / chi_n < (1.4 + 2 / (n + 1)) * cc)
                if hsig:
                    pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * ys
                else:
                    pc = (1 - cs) * pc
                
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    zi = (offspring[i][1] - old_mean) / sigma
                    rank_mu += weights[i] * np.outer(zi, zi)
                    
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * C) + cmu * rank_mu
                
                # Ensure symmetry and positive semi-definiteness
                C = (C + C.T) / 2.0
                try:
                    C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    C_sqrt = np.eye(n)
                
                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                
                # Convergence criteria
                if sigma < 1e-12:
                    break
                if no_improve > 15 and sigma < 0.2:
                    break

        # Phase 1: Global Exploration with Differential Evolution
        de_budget = int(budget * 0.6)
        run_de(de_budget)
        
        if check_deadline():
            return best_x, best_f
        
        # Phase 2: Local Refinement with CMA-ES
        if best_x is not None:
            cmaes_budget = budget - calls_made
            if cmaes_budget > 0:
                # Initialize CMA-ES around the best found by DE
                # Use a moderate sigma to allow some exploration but focus on refinement
                initial_sigma = 1.0 
                run_cmaes(best_x.copy(), initial_sigma, cmaes_budget)
        
        return best_x, best_f

# EVOLVE-BLOCK-END
