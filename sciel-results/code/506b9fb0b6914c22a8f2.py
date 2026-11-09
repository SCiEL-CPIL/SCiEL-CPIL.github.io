import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer for BBOB f15 (Rastrigin rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270  # 30s safety margin

        rng = np.random.default_rng(42)

        # DE parameters
        pop_size = 20
        F = 0.8  # Mutation factor
        CR = 0.5  # Crossover rate

        # Initialize population uniformly in [-5, 5]^dim
        population = rng.uniform(-5.0, 5.0, size=(pop_size, dim))
        fitnesses = np.array([func(x) for x in population])
        calls = pop_size

        best_idx = np.argmin(fitnesses)
        best_x = population[best_idx].copy()
        best_f = fitnesses[best_idx]

        # Main DE loop
        while calls < budget and time.time() < deadline:
            new_pop = population.copy()
            new_fitnesses = fitnesses.copy()

            for i in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break

                # Select three distinct random indices different from i
                while True:
                    a, b, c = rng.choice(pop_size, size=3, replace=False)
                    if a != b and b != c and a != c and a != i and b != i and c != i:
                        break

                # Mutation: v = x_a + F * (x_b - x_c)
                v = population[a] + F * (population[b] - population[c])
                v = np.clip(v, -5.0, 5.0)

                # Crossover: binary crossover with probability CR
                r = rng.random(dim)
                j_rand = rng.randint(dim)
                trial = population[i].copy()
                mask = (r <= CR) | (np.arange(dim) == j_rand)
                trial[mask] = v[mask]
                trial = np.clip(trial, -5.0, 5.0)

                # Evaluate trial
                ft = func(trial)
                calls += 1

                # Selection
                if ft <= fitnesses[i]:
                    new_pop[i] = trial
                    new_fitnesses[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()

            population = new_pop
            fitnesses = new_fitnesses

            # Check for convergence: if all fitnesses are very close, restart
            if np.std(fitnesses) < 1e-8:
                # Keep best, re-randomize others
                population = np.vstack([best_x.reshape(1, -1)] + 
                                       [rng.uniform(-5.0, 5.0, size=(1, dim)) for _ in range(pop_size - 1)])
                fitnesses = np.array([best_f] + [func(x) for x in population[1:]])
                calls += pop_size - 1

        return best_x, best_f
