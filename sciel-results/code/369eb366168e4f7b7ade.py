"""Combined optimizer for BBOB f23 (Katsuura): Multi-start Nelder-Mead + Adaptive (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive ES refinement for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget
        deadline = time.time() + 280

        best_x = None
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return x, fx

        rng = np.random.default_rng(12345)

        # Phase 1: Differential Evolution (DE/rand/1)
        # Katsuura is highly multimodal; DE is more robust than NM for global exploration.
        pop_size = 50
        # Reserve budget for Phase 2 and 3
        # Phase 2 uses ~5 candidates * 20 calls = 100 calls
        # Phase 3 uses ~50 calls
        # So reserve ~150 calls.
        # DE uses the rest.
        de_budget = B - 150
        
        # Initialize population uniformly in [-5, 5]
        population = rng.uniform(-5.0, 5.0, (pop_size, d))
        pop_f = np.empty(pop_size)
        
        # Evaluate initial population
        for i in range(pop_size):
            if calls >= de_budget or time.time() > deadline - 30:
                break
            x, fx = safe_call(population[i])
            population[i] = x
            pop_f[i] = fx
        
        # DE Parameters
        F = 0.8  # Scaling factor
        CR = 0.9  # Crossover rate
        
        # Track best candidates for refinement
        local_minima = []
        
        # Main DE loop
        # Each generation consumes pop_size calls.
        # We can run for at most de_budget // pop_size generations.
        max_generations = de_budget // pop_size
        
        for gen in range(max_generations):
            if calls >= de_budget or time.time() > deadline - 30:
                break
                
            # Sort population by fitness to keep track of best
            order = np.argsort(pop_f)
            population = population[order]
            pop_f = pop_f[order]
            
            # Update global best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = population[0].copy()
            
            # Add current best to local_minima if it's distinct enough (optional, but let's just collect top few at end)
            
            for i in range(pop_size):
                if calls >= de_budget or time.time() > deadline - 30:
                    break
                    
                # Select 3 distinct random individuals
                # Ensure they are distinct from i and each other
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                # Mutation: v = x[r1] + F * (x[r2] - x[r3])
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover: u = v if rand < CR or j == j_rand else x[i]
                j_rand = rng.integers(d)
                u = np.empty(d)
                for j in range(d):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = population[i][j]
                
                u = np.clip(u, -5.0, 5.0)
                u, fu = safe_call(u)
                
                # Selection
                if fu <= pop_f[i]:
                    population[i] = u
                    pop_f[i] = fu

        # Collect top candidates from DE population for refinement
        order = np.argsort(pop_f)
        top_k = min(5, pop_size)
        for k in range(top_k):
            local_minima.append((pop_f[order[k]], population[order[k]].copy()))

        # Phase 2: Adaptive (1+1)-ES refinement on top candidates
        # Sort local minima and refine the best ones
        local_minima.sort(key=lambda t: t[0])
        
        reserve_budget = 30
        available_for_es = B - calls - reserve_budget
        
        if available_for_es > 0:
            calls_per_candidate = max(10, available_for_es // max(1, top_k))
            
            for k in range(top_k):
                if calls >= B - 20 or time.time() > deadline - 15:
                    break
                
                fx_init, x_init = local_minima[k]
                
                # Adaptive (1+1)-ES with success-rate-based step adaptation
                x = x_init.copy()
                fx = fx_init
                sigma = 0.3  # Start with moderate step size
                
                window_size = 8
                successes = []
                
                max_steps = calls_per_candidate
                
                for _ in range(max_steps):
                    if calls >= B - 20 or time.time() > deadline - 10:
                        break
                    
                    y = x + sigma * rng.standard_normal(d)
                    y = np.clip(y, -5.0, 5.0)
                    y, fy = safe_call(y)
                    
                    if fy < fx:
                        successes.append(1)
                        x = y
                        fx = fy
                        sigma *= 1.08  # Increase step on success
                    else:
                        successes.append(0)
                        sigma *= 0.92  # Decrease step on failure
                    
                    if len(successes) > window_size:
                        successes.pop(0)
                    
                    # Adapt sigma based on success rate
                    if len(successes) == window_size:
                        sr = sum(successes) / window_size
                        if sr > 0.45:
                            sigma *= 1.03
                        elif sr < 0.15:
                            sigma *= 0.97
                    
                    sigma = np.clip(sigma, 1e-5, 3.0)

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 3: Final fine-tuning with very small adaptive steps
        if best_x is not None and calls < B - 5 and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.02
            
            while calls < B - 3 and time.time() < deadline - 3:
                y = np.clip(x + sigma * rng.standard_normal(d), -5.0, 5.0)
                y, fy = safe_call(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.01
                else:
                    sigma *= 0.96
                    if sigma < 1e-7:
                        break

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
