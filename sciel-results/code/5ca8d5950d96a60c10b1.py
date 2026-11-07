import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290
        
        best_x = None
        best_f = float('inf')
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Population-based Exploration
        pop_size = 10
        sigma = 1.5
        rng = np.random.default_rng(42)
        
        # Initialize population
        population = [np.zeros(dim)]
        for _ in range(pop_size - 1):
            population.append(rng.uniform(-4, 4, dim))
            
        pop_f = []
        for x in population:
            if calls >= budget or time.time() > deadline:
                break
            pop_f.append(eval_func(x))
            
        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Evolve population for a few generations
        # We want to spend ~50-100 calls here.
        # If pop_size=10, and we generate 5 new individuals per generation, that's 5 calls/gen.
        # 10 generations = 50 calls.
        max_gens = 10
        for gen in range(max_gens):
            if calls >= budget or time.time() > deadline:
                break
            
            # Sort population
            indices = np.argsort(pop_f)
            population = [population[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]
            
            # Keep top half, generate new half
            half = pop_size // 2
            new_population = population[:half].copy()
            new_f = pop_f[:half].copy()
            
            for _ in range(pop_size - half):
                if calls >= budget or time.time() > deadline:
                    break
                parent_idx = rng.integers(0, half)
                parent = population[parent_idx]
                child = parent + sigma * rng.standard_normal(dim)
                child = np.clip(child, -5, 5)
                new_population.append(child)
                
            # Evaluate new individuals
            for x in new_population[half:]:
                if calls >= budget or time.time() > deadline:
                    break
                new_f.append(eval_func(x))
                
            population = new_population
            pop_f = new_f
            
            # Adapt sigma
            # Simple adaptation: if best improved, increase sigma slightly, else decrease
            # This is a heuristic.
            if gen > 0:
                # Check if the best of the new population is better than the best of the old top half
                # Actually, just use a fixed schedule or simple rule.
                sigma *= 0.98 # Slowly decrease to converge
        
        # Phase 2: (1+1)-ES Exploitation
        # Start from the best point found in Phase 1
        best_idx = np.argmin(pop_f)
        x = population[best_idx].copy()
        f = pop_f[best_idx]
        
        sigma_es = 0.5 # Start with smaller step for exploitation
        max_es_calls = budget - calls - 50 # Reserve 50 for polish
        
        for _ in range(max_es_calls):
            if calls >= budget or time.time() > deadline:
                break
            
            y = x + sigma_es * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = eval_func(y)
            
            if fy < f:
                x = y
                f = fy
                sigma_es *= 1.02 # Increase step on success
            else:
                sigma_es *= 0.95 # Decrease step on failure
                
            # Clamp sigma
            sigma_es = np.clip(sigma_es, 0.01, 5.0)
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
        # Phase 3: Coordinate-wise Polish
        if calls < budget and time.time() < deadline and best_x is not None:
            x = best_x.copy()
            f = best_f
            step = 0.1
            for _ in range(50):
                if calls >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    for sign in [1, -1]:
                        if calls >= budget or time.time() > deadline:
                            break
                        y = x.copy()
                        y[d] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = eval_func(y)
                        if fy < f:
                            x = y
                            f = fy
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 0.001:
                        break
                else:
                    # Keep step or slightly increase if we are making progress?
                    # Keep step for stability
                    pass
            
            if f < best_f:
                best_f = f
                best_x = x.copy()

        return best_x, best_f
