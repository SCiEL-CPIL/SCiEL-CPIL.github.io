"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer for Büche-Rastrigin f4 on [-5,5]^5.
    
    Combines:
    1. Structured exploration (integer lattice + random) to find good basins
    2. Population-based (μ+λ)-ES with adaptive step size for robust local refinement
    3. Final high-precision polish pass
    
    The Büche-Rastrigin function has periodic local minima. After hidden 
    rotation/shift, the structure is preserved but transformed. We exploit 
    this by first sampling integer lattice points (likely near local minima 
    in the original frame), then using adaptive population-based local search 
    to refine.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline

        # Phase 1: Structured Exploration
        # Sample integer lattice points and random continuous points
        # to identify good basins for local refinement
        
        n_explore = min(60, budget // 3)
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        budget_used = 0
        
        # Evaluate integer lattice points (likely near local minima)
        n_lattice = n_explore // 2
        for _ in range(n_lattice):
            if budget_used >= budget or not time_ok():
                break
            pt = rng.integers(-4, 5, size=dim).astype(float)
            pt = np.clip(pt, -5, 5)
            f_val = func(pt)
            budget_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()
        
        # Evaluate random continuous points
        n_random = n_explore - n_lattice
        for _ in range(n_random):
            if budget_used >= budget or not time_ok():
                break
            pt = np.clip(rng.uniform(-5, 5, dim), -5, 5)
            f_val = func(pt)
            budget_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()
        
        # Phase 2: Population-based (μ+λ)-ES with Adaptive Step Size
        # Use a small population for robustness and diversity
        
        mu = 4
        lam = 8
        sigma = 1.0
        min_sigma = 1e-4
        max_sigma = 5.0
        
        # Initialize population around the best point found
        pop = [best_x.copy() + sigma * rng.standard_normal(dim) for _ in range(mu)]
        for i in range(len(pop)):
            pop[i] = np.clip(pop[i], -5, 5)
        
        # Evaluate initial population
        pop_f = []
        for i in range(mu):
            if budget_used >= budget or not time_ok():
                break
            f_val = func(pop[i])
            budget_used += 1
            pop_f.append(f_val)
            if f_val < best_f:
                best_f = f_val
                best_x = pop[i].copy()
        
        # Trim to those actually evaluated
        mu_actual = len(pop_f)
        if mu_actual == 0:
            return best_x, best_f
        pop = pop[:mu_actual]
        
        # Track sigma adaptation
        success_count = 0
        attempts = 0
        
        while budget_used < budget and time_ok():
            # Generate offspring
            offspring = []
            offspring_f = []
            
            for i in range(lam):
                if budget_used >= budget or not time_ok():
                    break
                # Select parent (tournament selection)
                if mu_actual > 1:
                    idx = rng.integers(0, mu_actual)
                else:
                    idx = 0
                
                # Mutation
                step = sigma * rng.standard_normal(dim)
                child = pop[idx].copy() + step
                child = np.clip(child, -5, 5)
                
                f_val = func(child)
                budget_used += 1
                offspring.append(child)
                offspring_f.append(f_val)
                attempts += 1
                
                if f_val < best_f:
                    best_f = f_val
                    best_x = child.copy()
                    success_count += 1
            
            if len(offspring) == 0:
                break
            
            # Combine parents and offspring
            combined = pop + offspring
            combined_f = pop_f + offspring_f
            
            # Select next generation
            if len(combined_f) >= mu_actual:
                # Sort by fitness
                order = np.argsort(combined_f)
                new_pop = [combined[i].copy() for i in order[:mu_actual]]
                new_pop_f = [combined_f[i] for i in order[:mu_actual]]
                pop = new_pop
                pop_f = new_pop_f
                mu_actual = len(pop)
            
            # Adapt sigma based on success rate
            if attempts >= 10:
                success_rate = success_count / attempts
                if success_rate > 0.3:
                    sigma *= 1.1
                elif success_rate < 0.1:
                    sigma *= 0.9
                sigma = max(min_sigma, min(sigma, max_sigma))
                success_count = 0
                attempts = 0
                
                # If converged, break to try next starting point
                if sigma < 0.01:
                    break
        
        # Phase 3: Final High-Precision Polish
        # If budget remains, do a fine-grained local search around the best point
        
        remaining = budget - budget_used
        if remaining > 10 and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.05  # Small step size for fine refinement
            min_sigma = 1e-5
            max_sigma = 0.5
            window_size = 5
            success_window = []
            
            for _ in range(remaining):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                budget_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    success_window.append(1)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    success_window.append(0)
                
                # Adapt sigma
                if len(success_window) >= window_size:
                    success_rate = np.mean(success_window[-window_size:])
                    if success_rate > 0.3:
                        sigma *= 1.1
                    elif success_rate < 0.1:
                        sigma *= 0.8
                    if len(success_window) > window_size:
                        success_window.pop(0)
                    sigma = max(min_sigma, min(sigma, max_sigma))
        
        return best_x, best_f
# EVOLVE-BLOCK-END
