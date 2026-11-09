"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with aggressive mutation for multimodal Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)

        # Population size - large enough for diversity, small enough to fit in budget
        # 500 / 25 = 20 generations
        lambda_ = 25

        # Initialize population uniformly in [-5, 5]^dim
        pop = rng.uniform(-5, 5, size=(lambda_, dim))
        pop_f = np.array([func(x) for x in pop])

        calls_made = lambda_
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        # DE parameters
        F = 0.8  # Mutation scale factor - aggressive
        CR = 0.5  # Crossover rate

        # Track stagnation for potential restart
        stagnation_count = 0
        prev_best_f = best_f

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate offspring for each individual
            offspring = np.empty_like(pop)
            offspring_f = np.empty(lambda_)

            for i in range(lambda_):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select three distinct random indices different from i
                candidates = list(range(lambda_))
                candidates.remove(i)
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

                # DE/rand/1 mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                v = np.clip(v, -5, 5)

                # Binomial crossover
                j_rand = rng.integers(0, dim)
                cross_mask = rng.random(dim) < CR
                cross_mask[j_rand] = True

                u = np.where(cross_mask, v, pop[i])
                u = np.clip(u, -5, 5)

                fu = func(u)
                calls_made += 1

                # Selection: keep better of parent and offspring
                if fu <= pop_f[i]:
                    offspring[i] = u
                    offspring_f[i] = fu
                else:
                    offspring[i] = pop[i]
                    offspring_f[i] = pop_f[i]

            # Update population
            pop = offspring
            pop_f = offspring_f

            # Track best
            new_best_idx = np.argmin(pop_f)
            if pop_f[new_best_idx] < best_f:
                best_f = pop_f[new_best_idx]
                best_x = pop[new_best_idx].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

            # If stagnating for too long, inject random individuals to maintain diversity
            if stagnation_count >= 5:
                n_reinject = max(1, lambda_ // 5)
                for _ in range(n_reinject):
                    if calls_made >= budget:
                        break
                    idx = rng.integers(0, lambda_)
                    pop[idx] = rng.uniform(-5, 5, size=dim)
                    pop_f[idx] = func(pop[idx])
                    calls_made += 1

                # Update best after re-injection
                new_best_idx = np.argmin(pop_f)
                if pop_f[new_best_idx] < best_f:
                    best_f = pop_f[new_best_idx]
                    best_x = pop[new_best_idx].copy()

                stagnation_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
