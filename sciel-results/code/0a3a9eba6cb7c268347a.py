"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive population-based optimizer for anisotropic BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 330s kill

        call_count = [0]
        best_x = [None]
        best_f = [np.inf]

        def wrapped(x):
            if call_count[0] >= budget:
                return best_f[0]
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            if fx < best_f[0]:
                best_f[0] = fx
                best_x[0] = x.copy()
            return fx

        rng = np.random.default_rng(12345)

        # Phase 1: Broad exploration with diverse population
        pop_size = 20
        sigma = 3.0  # initial step size
        center = np.zeros(dim)
        
        # Initialize population around center with some spread
        population = []
        for i in range(pop_size):
            if i == 0:
                x = center.copy()
            else:
                x = center + sigma * rng.standard_normal(dim)
            x = np.clip(x, -5.0, 5.0)
            population.append((x, wrapped(x)))
        
        population.sort(key=lambda t: t[1])
        
        # Phase 2: Adaptive local search with shrinking step size
        # Use remaining budget for iterative refinement
        remaining = budget - call_count[0]
        max_iterations = 50
        
        for iteration in range(max_iterations):
            if call_count[0] >= budget:
                break
            if time.time() > deadline:
                break
            
            # Adaptive sigma: shrink if we're making progress, expand if stuck
            best_in_pop = population[0]
            
            # Generate offspring around the best few individuals
            n_offspring = min(15, budget - call_count[0])
            if n_offspring <= 0:
                break
            
            candidates = []
            for j in range(n_offspring):
                # Pick a parent from top 3
                parent_idx = min(j % 3, len(population) - 1)
                parent_x = population[parent_idx][0]
                
                # Adaptive step: start large, shrink over iterations
                current_sigma = sigma * (0.95 ** iteration)
                child_x = parent_x + current_sigma * rng.standard_normal(dim)
                child_x = np.clip(child_x, -5.0, 5.0)
                child_f = wrapped(child_x)
                candidates.append((child_x, child_f))
            
            # Merge with existing population and keep top pop_size
            population.extend(candidates)
            population.sort(key=lambda t: t[1])
            population = population[:pop_size]
            
            # Update sigma based on improvement
            if best_f[0] < population[0][1]:
                pass  # No change needed
            else:
                # Check if we've made progress in this iteration
                prev_best = population[0][1] if len(population) > 0 else np.inf
                if candidates[0][1] < prev_best:
                    sigma *= 0.8  # shrink when making progress
                else:
                    sigma *= 1.2  # expand when stuck
                    sigma = min(sigma, 4.0)  # cap expansion
        
        # Phase 3: Final fine-tuning with very small steps around best
        if best_x[0] is not None and call_count[0] < budget:
            fine_sigma = 0.1
            for _ in range(30):
                if call_count[0] >= budget:
                    break
                if time.time() > deadline:
                    break
                x = best_x[0] + fine_sigma * rng.standard_normal(dim)
                x = np.clip(x, -5.0, 5.0)
                wrapped(x)
                fine_sigma *= 0.8
        
        if best_x[0] is None:
            # Fallback: must have made at least one call
            x0 = np.zeros(dim)
            call_count[0] = 1
            best_x[0] = x0.copy()
            best_f[0] = float(func(x0))

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
