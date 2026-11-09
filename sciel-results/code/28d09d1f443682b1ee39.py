"""Rewritten optimizer: differential-evolution-style population search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with population, aggressive mutation, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Population size
        lam = 20
        F = 0.8  # mutation scale
        CR = 0.5  # crossover rate

        # Initialize population
        pop = rng.uniform(-5, 5, (lam, dim))
        fvals = np.array([func(x) for x in pop])

        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]
        calls_made = lam

        # Track stagnation
        stagnation_counter = 0
        no_improve_window = 40  # generations without improvement triggers restart

        # Adaptive sigma for local refinement (hybrid)
        sigma = 1.0

        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break

            # Sort population by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]

            new_pop = [pop[0].copy()]  # elitism: keep best
            new_fvals = [fvals[0]]

            for i in range(1, lam):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select three distinct parents different from i
                candidates = list(range(lam))
                candidates.remove(i)
                if len(candidates) >= 3:
                    a, b, c = rng.choice(candidates, size=3, replace=False)
                else:
                    a, b, c = rng.choice(lam, size=3, replace=True)

                # Mutation: DE/rand/1
                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, -5, 5)

                # Crossover: binomial
                trial = v.copy()
                jrand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]

                # Evaluate
                ft = func(trial)
                calls_made += 1

                # Selection
                if ft <= fvals[i]:
                    new_pop.append(trial)
                    new_fvals.append(ft)
                else:
                    new_pop.append(pop[i].copy())
                    new_fvals.append(fvals[i])

            pop = np.array(new_pop)
            fvals = np.array(new_fvals)

            # Update best
            new_best_idx = np.argmin(fvals)
            new_best_f = fvals[new_best_idx]
            if new_best_f < best_f - 1e-10:
                best_f = new_best_f
                best_x = pop[new_best_idx].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # If stagnated, inject diversity by restarting some individuals
            if stagnation_counter >= no_improve_window:
                # Replace worst half with random points
                n_replace = lam // 2
                worst_indices = np.argsort(-fvals)[:n_replace]
                for idx in worst_indices:
                    if calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break
                    new_x = rng.uniform(-5, 5, dim)
                    new_f = func(new_x)
                    calls_made += 1
                    pop[idx] = new_x
                    fvals[idx] = new_f

                # Re-check best
                new_best_idx = np.argmin(fvals)
                new_best_f = fvals[new_best_idx]
                if new_best_f < best_f - 1e-10:
                    best_f = new_best_f
                    best_x = pop[new_best_idx].copy()
                    stagnation_counter = 0

                # Also do a few local refinements around best
                for _ in range(5):
                    if calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break
                    y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y
                        sigma *= 1.2
                    else:
                        sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
