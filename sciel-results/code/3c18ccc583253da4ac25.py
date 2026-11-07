"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution for global search followed by (1+1)-ES for local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        dim = self.dim
        calls = 0
        max_calls = self.budget

        def eval_func(x):
            nonlocal calls
            calls += 1
            return func(x)

        def check_time():
            return time.time() < deadline - 3.0

        def check_budget():
            return calls < max_calls

        # Phase 1: Differential Evolution
        pop_size = 20
        F = 0.8  # differential weight
        CR = 0.9  # crossover rate

        # Initialize population
        population = np.random.uniform(-5, 5, (pop_size, dim))
        # Include origin and some structured points
        population[0] = np.zeros(dim)
        for i in range(1, 5):
            population[i] = np.random.uniform(-4, 4, dim)

        fitness = np.array([eval_func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]

        # DE main loop
        max_de_generations = 15  # ~300 calls for DE phase
        for gen in range(max_de_generations):
            if not check_time() or not check_budget():
                break

            for i in range(pop_size):
                if not check_time() or not check_budget():
                    break

                # Select three distinct random indices
                r1, r2, r3 = np.random.choice(pop_size - 1, 3, replace=False)
                if r1 >= i:
                    r1 += 1
                if r2 >= i:
                    r2 += 1
                if r3 >= i:
                    r3 += 1
                if r2 == r3:
                    r3 = (r3 + 1) % pop_size
                    if r3 == i:
                        r3 = (r3 + 1) % pop_size

                # Create trial vector using DE/rand/1
                v = population[r1] + F * (population[r2] - population[r3])
                v = np.clip(v, -5, 5)

                # Binary crossover
                trial = population[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]

                # Selection
                f_trial = eval_func(trial)
                if f_trial <= fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

        # Phase 2: Local refinement with (1+1)-ES
        if check_time() and check_budget():
            x = best_x.copy()
            fx = best_f
            sigma = 0.5

            while check_time() and check_budget():
                y = np.clip(x + sigma * np.random.standard_normal(dim), -5, 5)
                fy = eval_func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                sigma = np.clip(sigma, 0.001, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
