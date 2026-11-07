"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid DE + (1+1)-ES for multimodal Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(123)
        d = self.dim
        budget = self.budget

        # Phase 1: Differential Evolution
        pop_size = 12
        F = 0.8
        CR = 0.9
        de_budget = int(budget * 0.7)

        pop = rng.uniform(-5, 5, (pop_size, d))
        fitness = np.array([func(pop[i]) for i in range(pop_size)])
        calls_used = pop_size

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_fx = fitness[best_idx]

        while calls_used < de_budget:
            if time.time() > deadline:
                break

            # Generate new population
            new_pop = []
            new_fitness = []
            for i in range(pop_size):
                # Select three distinct random individuals
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(candidates, 3, replace=False)

                # Mutation
                v = pop[a] + F * (pop[b] - pop[c])

                # Crossover
                r = rng.random(d)
                j_rand = rng.integers(d)
                trial = np.where(r < CR, v, pop[i])
                trial[j_rand] = v[j_rand]

                # Clip to bounds
                trial = np.clip(trial, -5, 5)

                # Evaluate
                f_trial = func(trial)
                calls_used += 1

                # Selection
                if f_trial <= fitness[i]:
                    new_pop.append(trial)
                    new_fitness.append(f_trial)
                else:
                    new_pop.append(pop[i])
                    new_fitness.append(fitness[i])

            pop = np.array(new_pop)
            fitness = np.array(new_fitness)

            # Update global best
            idx = np.argmin(fitness)
            if fitness[idx] < best_fx:
                best_fx = fitness[idx]
                best_x = pop[idx].copy()

        # Phase 2: (1+1)-ES refinement from best point
        x = best_x.copy()
        fx = best_fx
        sigma = 0.5
        no_improve = 0

        while calls_used < budget:
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                x = y.copy()
                fx = fy
                sigma *= 1.1
                no_improve = 0
            else:
                sigma *= 0.85
                no_improve += 1

            if no_improve > 20:
                # Try a larger jump
                big_step = 1.5 * rng.standard_normal(d)
                y2 = np.clip(x + big_step, -5, 5)
                fy2 = func(y2)
                calls_used += 1
                if fy2 < fx:
                    x = y2.copy()
                    fx = fy2
                    sigma = 1.0
                    no_improve = 0
                else:
                    no_improve = 0
                    sigma = max(sigma * 0.7, 0.05)

            sigma = np.clip(sigma, 0.02, 3.0)

        return x, fx
# EVOLVE-BLOCK-END
