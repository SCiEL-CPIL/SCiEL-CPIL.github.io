"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """Differential Evolution + Nelder-Mead optimizer for Katsuura-style multimodal function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        best_x = np.zeros(n)
        best_f = float('inf')
        total_calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, total_calls
            x = np.clip(x, -5, 5)
            y = func(x)
            total_calls += 1
            if y < best_f:
                best_f = y
                best_x = x.copy()
            return y

        # Phase 1: Differential Evolution for global search
        # DE is known to be effective for multimodal functions like Katsuura
        
        def differential_evolution(x0, max_calls, rng_local):
            nonlocal best_x, best_f, total_calls
            
            # DE parameters
            pop_size = 10  # Population size
            F = 0.8        # Differential weight
            CR = 0.9       # Crossover rate
            
            # Initialize population
            pop = [x0.copy() + rng_local.normal(0, 2.0, n) for _ in range(pop_size)]
            pop = [np.clip(p, -5, 5) for p in pop]
            
            # Evaluate initial population
            pop_f = np.array([evaluate(p) for p in pop])
            
            # Main DE loop
            while total_calls < budget and time.time() < deadline:
                # Sort by fitness
                order = np.argsort(pop_f)
                pop = [pop[i] for i in order]
                pop_f = pop_f[order]
                
                # Update global best
                if pop_f[0] < best_f:
                    best_f = pop_f[0]
                    best_x = pop[0].copy()
                
                # Generate trial vectors
                for i in range(pop_size):
                    if total_calls >= budget or time.time() > deadline:
                        return
                    
                    # Select three distinct random individuals
                    while True:
                        a, b, c = rng_local.choice(pop_size, 3, replace=False)
                        if len(set([i, a, b, c])) == 4:
                            break
                    
                    # Differential variation
                    v = pop[a] + F * (pop[b] - pop[c])
                    v = np.clip(v, -5, 5)
                    
                    # Binomial crossover
                    r = rng_local.random()
                    jrand = rng_local.integers(n)
                    trial = pop[i].copy()
                    for j in range(n):
                        if r < CR or j == jrand:
                            trial[j] = v[j]
                    
                    # Evaluate trial
                    trial_f = evaluate(trial)
                    
                    # Selection
                    if trial_f <= pop_f[i]:
                        pop[i] = trial
                        pop_f[i] = trial_f

        # Phase 2: Nelder-Mead refinement for local precision
        
        def nelder_mead(x0, max_calls, rng_local):
            nonlocal best_x, best_f, total_calls
            
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(n):
                x_temp = x0.copy()
                if abs(x0[i]) > 1e-10:
                    x_temp[i] += 0.1 * x0[i]
                else:
                    x_temp[i] += 0.1
                x_temp = np.clip(x_temp, -5, 5)
                simplex.append(x_temp)
            
            # Evaluate simplex
            simplex_f = [evaluate(s) for s in simplex]
            
            # Nelder-Mead parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            # Main NM loop
            while total_calls < budget and time.time() < deadline:
                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Update global best
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()
                
                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                
                # Reflection
                x_ref = centroid + alpha * (centroid - worst)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = evaluate(x_ref)
                
                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = evaluate(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < simplex_f[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                        f_con = evaluate(x_con)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (centroid - worst)
                        f_con = evaluate(x_con)
                    
                    x_con = np.clip(x_con, -5, 5)
                    f_con = evaluate(x_con)
                    
                    if f_con < min(f_ref, simplex_f[-1]):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = evaluate(simplex[i])
                
                # Check convergence
                if np.max(simplex_f) - np.min(simplex_f) < 1e-10:
                    break

        # Phase 1: Initial random sampling to find diverse starting points
        n_init = max(10, budget // 10)
        X_init = rng.uniform(-5, 5, (n_init, n))
        y_init = np.array([evaluate(x) for x in X_init])
        
        # Select top 3 diverse points for DE
        top_k = min(3, n_init)
        best_indices = np.argsort(y_init)[:top_k]
        
        # Phase 2: Run DE from each top point
        remaining_budget = budget - total_calls
        if remaining_budget > 0:
            budget_per_de = remaining_budget // (top_k + 1)
            
            for i in range(top_k):
                if total_calls >= budget or time.time() > deadline:
                    break
                
                x_start = X_init[best_indices[i]].copy()
                sub_rng = np.random.default_rng(42 + i)
                differential_evolution(x_start, budget_per_de, sub_rng)
            
            # Phase 3: Nelder-Mead refinement from the best point found by DE
            if total_calls < budget and time.time() < deadline:
                nm_budget = budget - total_calls
                nm_rng = np.random.default_rng(99)
                nelder_mead(best_x.copy(), nm_budget, nm_rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
