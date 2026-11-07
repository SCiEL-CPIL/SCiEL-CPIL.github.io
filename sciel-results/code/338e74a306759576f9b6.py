"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function using Differential Evolution."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution (DE/rand/1/bin) with adaptive F and restarts for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # DE parameters
        pop_size = 20
        F_min, F_max = 0.3, 0.9
        CR = 0.7

        # Initialize population uniformly in [-5, 5]^dim
        population = rng.uniform(-5, 5, size=(pop_size, dim))
        fitness = np.array([func(x) for x in population])
        calls_made = pop_size

        best_idx = np.argmin(fitness)
        global_best_x = population[best_idx].copy()
        global_best_fx = fitness[best_idx]

        # Adaptive F tracking
        F = 0.7
        f_success = 0
        f_trials = 0

        # Stagnation tracking
        stagnation_count = 0
        stagnation_limit = 25
        best_fx_history = global_best_fx

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate offspring for each member
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Pick three distinct random indices different from i
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)

                # Mutation: DE/rand/1
                v = population[r1] + F * (population[r2] - population[r3])

                # Crossover: binomial
                j_rand = rng.integers(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = population[i][j]

                trial = np.clip(trial, -5, 5)
                trial_fx = func(trial)
                calls_made += 1

                # Selection
                if trial_fx <= fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fx
                    f_success += 1
                    if trial_fx < global_best_fx:
                        global_best_fx = trial_fx
                        global_best_x = trial.copy()
                        stagnation_count = 0
                else:
                    f_trials += 1

                if calls_made >= budget:
                    break

            # Adapt F based on success rate
            if f_trials + f_success > 0:
                success_rate = f_success / (f_trials + f_success)
                F = np.clip(F + 0.1 * (success_rate - 0.5), F_min, F_max)
            f_success = 0
            f_trials = 0

            # Check stagnation
            if global_best_fx == best_fx_history:
                stagnation_count += 1
            else:
                best_fx_history = global_best_fx
                stagnation_count = 0

            # Restart if stagnated
            if stagnation_count >= stagnation_limit:
                # Replace worst half of population with random points
                n_replace = pop_size // 2
                worst_indices = np.argsort(fitness)[-n_replace:]
                for idx in worst_indices:
                    if calls_made >= budget:
                        break
                    new_x = rng.uniform(-5, 5, size=dim)
                    new_fx = func(new_x)
                    calls_made += 1
                    population[idx] = new_x
                    fitness[idx] = new_fx
                    if new_fx < global_best_fx:
                        global_best_fx = new_fx
                        global_best_x = new_x.copy()

                stagnation_count = 0
                F = 0.8  # Increase exploration on restart

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
