"""Optimized Katsuura f23 solver: DE/rand/1 with Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with Nelder-Mead polish for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit

        calls_made = 0
        best_x = None
        best_f = float('inf')

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # DE parameters
        pop_size = 50
        F = 0.8  # mutation scale
        CR = 0.9  # crossover rate
        max_de_evals = 350  # Reserve some budget for Nelder-Mead

        # Initialize population
        population = np.random.uniform(-4.5, 4.5, (pop_size, dim))
        fitness = np.array([evaluate(x) for x in population])

        # Track best in population
        best_idx = np.argmin(fitness)
        pop_best_x = population[best_idx].copy()
        pop_best_f = fitness[best_idx]

        # DE loop
        de_evals = 0
        while calls_made < max_de_evals and time.time() < deadline:
            for i in range(pop_size):
                if calls_made >= max_de_evals or time.time() >= deadline:
                    break

                # Select three distinct random indices
                while True:
                    r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                    if r1 != i and r2 != i and r3 != i:
                        break

                # Mutation: DE/rand/1
                v = population[r1] + F * (population[r2] - population[r3])

                # Crossover
                j_rand = np.random.randint(dim)
                trial = np.copy(population[i])
                for j in range(dim):
                    if np.random.random() < CR or j == j_rand:
                        trial[j] = v[j]

                # Evaluate trial
                trial = np.clip(trial, -5.0, 5.5)
                f_trial = evaluate(trial)
                de_evals += 1

                # Selection
                if f_trial <= fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < pop_best_f:
                        pop_best_f = f_trial
                        pop_best_x = trial.copy()

        # Nelder-Mead polish on the best point
        if calls_made < budget and time.time() < deadline:
            nm_x = pop_best_x.copy()
            nm_f = pop_best_f
            nm_simplex = [nm_x.copy()]
            for d in range(dim):
                perturbed = nm_x.copy()
                perturbed[d] += 0.1 if perturbed[d] < 4.9 else -0.1
                nm_simplex.append(perturbed)

            nm_fitness = [nm_f] + [evaluate(s) for s in nm_simplex[1:]]
            nm_iter = 0
            max_nm_iter = 30

            while nm_iter < max_nm_iter and calls_made < budget and time.time() < deadline:
                # Sort simplex by fitness
                order = np.argsort(nm_fitness)
                nm_simplex = [nm_simplex[i] for i in order]
                nm_fitness = [nm_fitness[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(nm_simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - nm_simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = evaluate(x_r)

                if f_r < nm_fitness[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - nm_simplex[-1])
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        nm_simplex[-1] = x_e
                        nm_fitness[-1] = f_e
                    else:
                        nm_simplex[-1] = x_r
                        nm_fitness[-1] = f_r
                elif f_r < nm_fitness[-2]:
                    nm_simplex[-1] = x_r
                    nm_fitness[-1] = f_r
                else:
                    # Contraction
                    rho = 0.5
                    x_c = centroid + rho * (nm_simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = evaluate(x_c)
                    if f_c < nm_fitness[-1]:
                        nm_simplex[-1] = x_c
                        nm_fitness[-1] = f_c
                    else:
                        # Shrink
                        for k in range(1, dim + 1):
                            nm_simplex[k] = nm_simplex[0] + 0.5 * (nm_simplex[k] - nm_simplex[0])
                            nm_fitness[k] = evaluate(nm_simplex[k])

                nm_iter += 1

                # Update global best
                for k in range(dim + 1):
                    if nm_fitness[k] < best_f:
                        best_f = nm_fitness[k]
                        best_x = nm_simplex[k].copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
