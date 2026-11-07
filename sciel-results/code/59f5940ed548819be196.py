import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        time_limit = 290.0

        best_x = None
        best_f = float('inf')
        calls = 0

        def record(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Initial exploration with Latin hypercube sampling for better coverage
        np.random.seed(42)
        explore_budget = min(50, budget // 5)
        
        # Latin Hypercube Sampling for better initial coverage
        if explore_budget >= n:
            for i in range(explore_budget):
                if time.time() - start_time > time_limit:
                    break
                # Generate LHS-like point: stratify each dimension
                x = np.zeros(n)
                for d in range(n):
                    stratum = i % (explore_budget // n + 1)
                    total_strata = explore_budget // n + 1
                    low = -5 + 10 * stratum / total_strata
                    high = -5 + 10 * (stratum + 1) / total_strata
                    x[d] = np.random.uniform(low, high)
                fx = func(x)
                calls += 1
                record(x, fx)
        else:
            for _ in range(explore_budget):
                if time.time() - start_time > time_limit:
                    break
                x = np.random.uniform(-5, 5, n)
                fx = func(x)
                calls += 1
                record(x, fx)

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls += 1

        # Phase 2: Differential Evolution with adaptive parameters
        # DE is more robust for black-box problems and handles constraints naturally
        pop_size = max(10, min(40, budget // 20))
        F = 0.8  # Mutation scale
        CR = 0.9  # Crossover rate
        adaptive_F = True
        
        # Initialize population around best point found
        pop = [best_x.copy() for _ in range(pop_size)]
        f_pop = [best_f for _ in range(pop_size)]
        
        # Diversify initial population
        for i in range(1, pop_size):
            if time.time() - start_time > time_limit:
                break
            if i == 1:
                # Second point: slightly perturbed best
                pop[i] = np.clip(best_x + 0.5 * np.random.randn(n), -5, 5)
            else:
                # Random points with bias toward center
                pop[i] = np.clip(np.random.uniform(-3, 3, n), -5, 5)
            f_pop[i] = func(pop[i])
            calls += 1
            record(pop[i], f_pop[i])
        
        # Sort population
        order = np.argsort(f_pop)
        pop = [pop[i] for i in order]
        f_pop = [f_pop[i] for i in order]
        
        no_improve_count = 0
        max_no_improve = 5
        
        while calls < budget and time.time() - start_time < time_limit:
            # Generate offspring
            new_pop = [p.copy() for p in pop]
            new_f = list(f_pop)
            
            for i in range(pop_size):
                if calls >= budget or time.time() - start_time > time_limit:
                    break
                    
                # Select two distinct random indices different from i
                r1, r2 = np.random.choice([j for j in range(pop_size) if j != i], 2, replace=False)
                
                # Mutation: differential mutation
                if adaptive_F:
                    # Adapt F based on progress
                    F_current = F * (0.8 + 0.4 * np.random.rand())
                else:
                    F_current = F
                
                mutant = np.clip(pop[r1] + F_current * (pop[r2] - pop[i]), -5, 5)
                
                # Crossover: binomial crossover
                trial = mutant.copy()
                j_rand = np.random.randint(n)
                for j in range(n):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = pop[i][j]
                
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls += 1
                record(trial, f_trial)
                
                # Selection
                if f_trial <= new_f[i]:
                    new_pop[i] = trial
                    new_f[i] = f_trial
            
            # Update population
            pop = new_pop
            f_pop = new_f
            
            # Sort
            order = np.argsort(f_pop)
            pop = [pop[i] for i in order]
            f_pop = [f_pop[i] for i in order]
            
            # Track improvement
            if f_pop[0] < best_f - 1e-10:
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Adaptive strategy: if stagnant, restart exploration
            if no_improve_count >= max_no_improve:
                # Perturb the best few individuals
                for i in range(min(5, pop_size)):
                    if time.time() - start_time > time_limit:
                        break
                    pop[i] = np.clip(pop[0] + np.random.randn(n) * 0.3, -5, 5)
                    f_pop[i] = func(pop[i])
                    calls += 1
                    record(pop[i], f_pop[i])
                
                no_improve_count = 0
                max_no_improve = min(max_no_improve + 2, 15)
                
                # Re-sort
                order = np.argsort(f_pop)
                pop = [pop[i] for i in order]
                f_pop = [f_pop[i] for i in order]
            
            # Convergence check
            if f_pop[0] - f_pop[-1] < 1e-15:
                break
        
        # Phase 3: Local refinement with Nelder-Mead simplex
        remaining = budget - calls
        if remaining > 10 and best_x is not None and time.time() - start_time < time_limit:
            from scipy.optimize import minimize
            
            # Use Nelder-Mead for local refinement
            x0 = best_x.copy()
            # Set initial simplex size based on problem scale
            initial_simplex = [x0 + 0.1 * np.eye(n)[i] for i in range(n + 1)]
            
            def bounded_func(x):
                x = np.clip(x, -5, 5)
                return func(x)
            
            try:
                result = minimize(bounded_func, x0, method='Nelder-Mead', 
                                  options={'maxiter': remaining // 2, 'xatol': 1e-6, 'fatol': 1e-8})
                # Count function calls made by scipy
                # We need to track this carefully
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass
        
        return best_x, best_f
