"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based Differential Evolution (DE/rand/1/bin) with adaptive parameters."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s deadline to stay well under 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Population size: 15 for good diversity without too many evaluations per generation
        pop_size = 15
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate

        # Initialize population uniformly in [-5, 5]^dim
        population = rng.uniform(-5, 5, (pop_size, dim))
        fitnesses = np.array([func(x) for x in population])
        calls_made = pop_size

        best_idx = int(np.argmin(fitnesses))
        best_x = population[best_idx].copy()
        best_f = fitnesses[best_idx]

        # Track for adaptive parameter control
        success_count = 0
        total_trials = 0
        F_adapt = F
        CR_adapt = CR

        generation = 0
        while calls_made < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Adaptive parameter adjustment
            if total_trials > 0:
                success_rate = success_count / total_trials
                # Adapt F based on success rate
                F_adapt = 0.5 + 0.5 * (1.0 - success_rate)
                F_adapt = np.clip(F_adapt, 0.4, 1.0)
                CR_adapt = 0.5 + 0.4 * success_rate
                CR_adapt = np.clip(CR_adapt, 0.6, 0.95)

            for i in range(pop_size):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select three distinct random individuals different from i
                while True:
                    r1 = int(rng.integers(pop_size))
                    if r1 != i:
                        break
                while True:
                    r2 = int(rng.integers(pop_size))
                    if r2 != i and r2 != r1:
                        break
                while True:
                    r3 = int(rng.integers(pop_size))
                    if r3 != i and r3 != r1 and r3 != r2:
                        break

                # Mutation: DE/rand/1
                v = population[r1] + F_adapt * (population[r2] - population[r3])
                v = np.clip(v, -5, 5)

                # Crossover: binomial
                j_rand = int(rng.integers(dim))
                trial = population[i].copy()
                for j in range(dim):
                    if rng.random() < CR_adapt or j == j_rand:
                        trial[j] = v[j]

                # Evaluate trial
                calls_made += 1
                f_trial = func(trial)
                total_trials += 1

                if f_trial <= fitnesses[i]:
                    population[i] = trial
                    fitnesses[i] = f_trial
                    success_count += 1
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        best_idx = i

            generation += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
