"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive parameters and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Initialize population
        pop_size = 20
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fvals = np.array([func(x) for x in pop])

        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]
        calls_made = pop_size

        # Adaptive parameters
        F = 0.8  # differential weight
        CR = 0.9  # crossover rate
        sigma = 1.0  # initial step size for local refinement

        # Track success for adaptation
        success_count = 0
        total_trials = 0
        window_size = 10

        for iteration in range(budget - pop_size):
            # Check deadline
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            # Adaptive parameter adjustment
            if total_trials > window_size:
                success_rate = success_count / total_trials
                # Adapt F and CR based on success rate
                F = np.clip(F + 0.1 * (success_rate - 0.5), 0.4, 1.0)
                CR = np.clip(CR + 0.05 * (success_rate - 0.5), 0.5, 1.0)
                success_count = 0
                total_trials = 0

            # Sort population by fitness
            order = np.argsort(fvals)
            sorted_pop = pop[order]
            sorted_fvals = fvals[order]

            # Generate new candidate using DE/rand/1
            # Select three distinct random individuals
            r1, r2, r3 = rng.choice(pop_size, 3, replace=False)

            # Differential mutation
            v = sorted_pop[r1] + F * (sorted_pop[r2] - sorted_pop[r3])
            v = np.clip(v, -5, 5)

            # Binary crossover
            j_rand = rng.integers(dim)
            trial = sorted_pop[0].copy()  # target is current best
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
            trial = np.clip(trial, -5, 5)

            # Evaluate trial
            f_trial = func(trial)
            calls_made += 1
            total_trials += 1

            # Selection
            if f_trial < sorted_fvals[0]:
                pop[order[0]] = trial
                fvals[order[0]] = f_trial
                success_count += 1

                # Update best
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
                    # Increase sigma on success for local refinement
                    sigma *= 1.2
            else:
                sigma *= 0.95

            # Local refinement around best point
            if iteration % 5 == 0 and calls_made < budget:
                # Try a small perturbation around the best point
                local_trial = best_x + sigma * rng.standard_normal(dim)
                local_trial = np.clip(local_trial, -5, 5)
                f_local = func(local_trial)
                calls_made += 1
                if f_local < best_f:
                    best_f = f_local
                    best_x = local_trial.copy()
                    sigma *= 1.3
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
