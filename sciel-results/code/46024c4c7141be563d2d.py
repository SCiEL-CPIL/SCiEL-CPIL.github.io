"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        # Differential Evolution parameters
        # Population size is typically 5-10 times the dimension
        pop_size = 10 * self.dim
        # Budget 500, pop_size 50 -> 10 generations
        # We will use a loop that respects the budget
        
        # Initialize population
        population = rng.uniform(-5, 5, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = pop_size
        
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_fx = fitness[best_idx]

        # DE Parameters
        F = 0.5  # Mutation scale factor
        CR = 0.5  # Crossover rate
        
        # Main DE loop
        while budget_used < self.budget:
            if time.time() >= deadline:
                break
            
            # Perform one generation
            for i in range(pop_size):
                if budget_used >= self.budget:
                    break
                if time.time() >= deadline:
                    break
                
                # Select three distinct random individuals
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                # Mutation
                mutant = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover
                trial = population[i].copy()
                j_rand = rng.integers(0, self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                f_trial = func(trial)
                budget_used += 1
                
                # Selection
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_fx:
                        best_x = trial.copy()
                        best_fx = f_trial

        # Multi-stage Nelder-Mead polish with decreasing step sizes
        remaining = self.budget - budget_used
        if remaining > 10 and time.time() < deadline - 5:
            try:
                from scipy.optimize import minimize
                polish_calls = [0]
                max_polish_calls = min(remaining, 200)

                def polish_func(z):
                    if polish_calls[0] >= max_polish_calls:
                        return best_fx
                    polish_calls[0] += 1
                    z_clipped = np.clip(z, -5, 5)
                    return func(z_clipped)

                # Stage 1: Medium-step Nelder-Mead
                initial_step = 0.5
                result = minimize(
                    polish_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_polish_calls // 2,
                        'initial_simplex': np.array([
                            best_x + initial_step * e
                            for e in np.eye(self.dim)
                        ]),
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if result.fun < best_fx:
                    best_x = np.clip(result.x, -5, 5)
                    best_fx = result.fun

                # Stage 2: Fine-step Nelder-Mead for high-precision polish
                remaining2 = max_polish_calls - polish_calls[0]
                if remaining2 > 5 and time.time() < deadline - 3:
                    initial_step2 = 0.05
                    result2 = minimize(
                        polish_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': remaining2,
                            'initial_simplex': np.array([
                                best_x + initial_step2 * e
                                for e in np.eye(self.dim)
                            ]),
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                    if result2.fun < best_fx:
                        best_x = np.clip(result2.x, -5, 5)
                        best_fx = result2.fun
            except Exception:
                pass  # If polish fails, keep the DE result

        return best_x, best_fx
# EVOLVE-BLOCK-END
