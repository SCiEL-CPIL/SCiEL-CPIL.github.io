"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Population-based Differential Evolution (DE) for Rosenbrock rotated."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        n_evals = 0
        max_evals = budget

        # Stratified initialization for better coverage
        pop_size = 20
        # Use Latin Hypercube-like stratified sampling
        pop = np.zeros((pop_size, dim))
        for d in range(dim):
            # Divide [-5, 5] into pop_size equal intervals, pick one point per interval
            edges = np.linspace(-5, 5, pop_size + 1)
            for i in range(pop_size):
                pop[i, d] = rng.uniform(edges[i], edges[i + 1])

        fvals = np.empty(pop_size)
        for i in range(pop_size):
            fvals[i] = func(pop[i])
            n_evals += 1
            if n_evals >= max_evals:
                break

        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        # Phase 1: Global exploration with DE/current-to-best/1
        # Phase 2: Local exploitation around best solution
        phase1_budget = int(0.7 * max_evals)

        # DE parameters
        F = 0.8
        CR = 0.7

        while n_evals < phase1_budget:
            best_idx = np.argmin(fvals)
            best_x = pop[best_idx].copy()
            best_f = fvals[best_idx]

            for i in range(pop_size):
                if n_evals >= phase1_budget:
                    break

                # DE/current-to-best/1: v = x_i + F*(best - x_i) + F*(x_r1 - x_r2)
                mask = np.ones(pop_size, dtype=bool)
                mask[i] = False
                mask[best_idx] = False
                candidates = np.where(mask)[0]
                if len(candidates) < 2:
                    candidates = np.where(mask)[0]
                    if len(candidates) < 2:
                        continue
                r1, r2 = rng.choice(candidates, size=2, replace=False)

                v = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])

                # Binomial crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                trial = np.clip(trial, -5, 5)
                f_trial = func(trial)
                n_evals += 1

                if f_trial < fvals[i]:
                    pop[i] = trial
                    fvals[i] = f_trial

                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()

            # Adaptive parameters
            progress = n_evals / phase1_budget
            F = 0.9 - 0.3 * progress
            CR = 0.8 - 0.2 * progress

        # Phase 2: Local refinement around best solution
        # Use small perturbations (Nelder-Mead-like or random local search)
        local_budget = max_evals - n_evals
        if local_budget > 0:
            # Create a local population around the best solution
            local_pop_size = min(10, local_budget)
            scale = 0.5  # Initial perturbation scale
            local_pop = np.zeros((local_pop_size, dim))
            local_fvals = np.empty(local_pop_size)

            local_pop[0] = best_x.copy()
            local_fvals[0] = best_f
            n_evals += 1  # Count the best_x evaluation? No, it was already evaluated.
            # Actually, we don't need to re-evaluate best_x, but let's be careful.
            # Let's not count it again. We'll start with 1 individual already known.
            n_evals -= 1  # Undo, we won't count the existing best

            for i in range(1, local_pop_size):
                if n_evals >= max_evals:
                    break
                local_pop[i] = np.clip(best_x + scale * rng.standard_normal(dim), -5, 5)
                local_fvals[i] = func(local_pop[i])
                n_evals += 1

            # Simple local DE with decreasing scale
            for gen in range(max(1, local_budget // local_pop_size)):
                if n_evals >= max_evals:
                    break

                local_best_idx = np.argmin(local_fvals)
                local_best_x = local_pop[local_best_idx].copy()
                local_best_f = local_fvals[local_best_idx]

                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()

                for i in range(local_pop_size):
                    if n_evals >= max_evals:
                        break

                    # DE/rand/1 within local population
                    if local_pop_size >= 3:
                        candidates = [j for j in range(local_pop_size) if j != i]
                        r1, r2 = rng.choice(candidates, size=2, replace=False)
                        v = local_pop[r1] + 0.5 * (local_pop[r2] - local_pop[i])
                    else:
                        v = local_pop[i] + 0.1 * rng.standard_normal(dim)

                    trial = np.clip(v, -5, 5)
                    f_trial = func(trial)
                    n_evals += 1

                    if f_trial < local_fvals[i]:
                        local_pop[i] = trial
                        local_fvals[i] = f_trial

                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

                # Decrease scale
                scale *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
