"""Optimized BBOB f6 solver using Differential Evolution with adaptive parameters."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer for BBOB f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        total_calls = 0
        rng = np.random.default_rng(42)

        # Helper to safely call func
        def safe_call(x):
            nonlocal total_calls
            if total_calls >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            total_calls += 1
            return fx

        # Differential Evolution
        pop_size = 10  # Small population for efficient exploration
        F = 0.5  # Scale factor
        CR = 0.9  # Crossover rate

        # Initialize population
        population = []
        for i in range(pop_size):
            if i == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4.0, 4.0, dim)
            fx = safe_call(x)
            if fx is None:
                break
            population.append((fx, x.copy()))

        if not population:
            # Fallback: just evaluate center
            x = np.zeros(dim)
            fx = safe_call(x)
            if fx is not None:
                return x, fx
            return np.zeros(dim), np.inf

        # Sort initial population
        population.sort(key=lambda t: t[0])
        best_f, best_x = population[0]

        # Adaptive parameters: adjust F and CR based on progress
        generation = 0
        no_improve = 0
        best_f_history = []

        while total_calls < budget and time.time() < deadline:
            generation += 1

            # Check for stagnation
            if best_f_history and abs(best_f - best_f_history[-1]) < 1e-10:
                no_improve += 1
            else:
                no_improve = 0
            best_f_history.append(best_f)
            if len(best_f_history) > 20:
                best_f_history.pop(0)

            # Adapt F and CR based on stagnation
            if no_improve > 10:
                F = min(F + 0.1, 1.0)
                CR = max(CR - 0.1, 0.5)
            elif no_improve > 5:
                F = min(F + 0.05, 0.8)
            else:
                F = max(F - 0.05, 0.3)
                CR = min(CR + 0.05, 0.95)

            # Create offspring for each individual in population
            new_population = []
            for i in range(pop_size):
                if total_calls >= budget or time.time() > deadline:
                    break

                # Select three distinct random individuals
                indices = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(indices, size=3, replace=False)

                # Mutation: DE/best/1 variant using best individual
                x_best = population[0][1]
                x_a = population[a][1]
                x_b = population[b][1]
                x_c = population[c][1]

                # Mutation vector
                v = x_best + F * (x_a - x_b) + F * (x_a - x_c)

                # Crossover
                trial = np.zeros(dim)
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = population[i][1][j]

                # Clip to bounds
                trial = np.clip(trial, -5.0, 5.0)

                # Evaluate
                ft = safe_call(trial)
                if ft is None:
                    break

                # Selection
                if ft <= population[i][0]:
                    new_population.append((ft, trial.copy()))
                else:
                    new_population.append(population[i])

            if not new_population:
                break

            # Update population
            population = new_population
            population.sort(key=lambda t: t[0])

            # Update best
            if population[0][0] < best_f:
                best_f = population[0][0]
                best_x = population[0][1].copy()
                no_improve = 0
            else:
                no_improve += 1

            # Early termination if converged
            if no_improve > 50 and best_f < 1.0:
                break

        # Final polish: local refinement around best point
        if total_calls < budget and time.time() < deadline and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(min(50, budget - total_calls)):
                if total_calls >= budget or time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.2, 0.5)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.8, 1e-4)
                    if sigma <= 1e-4:
                        break

        # Ensure valid result
        if best_x is None:
            best_x = np.zeros(dim)
            fx = safe_call(best_x)
            if fx is not None:
                best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
