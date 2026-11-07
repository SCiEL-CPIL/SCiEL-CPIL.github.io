"""Multi-start Differential Evolution for Katsuura f23."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """DE/rand/1 with population of 50, multi-start diversity, hybrid adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        pop_size = 50

        # Initialize population
        pop = np.zeros((pop_size, dim))
        # Structured starts to ensure immediate coverage
        pop[0] = 0.0
        pop[1] = rng.uniform(-1, 1, dim)
        # Add some grid-like points for better initial spread
        for i in range(2, 6):
            pop[i] = rng.uniform(-4, 4, dim)
        for i in range(6, pop_size):
            pop[i] = rng.uniform(-5, 5, dim)

        # Evaluate initial population
        fvals = np.array([func(p) for p in pop])
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        calls_used = pop_size

        # DE parameters
        F = 0.8
        CR = 0.9

        # Track for adaptive strategies
        no_improve = 0
        gen = 0

        while calls_used < budget:
            # Sort population by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]
            best_x = pop[0].copy()
            best_f = fvals[0]

            # Adaptive F and CR based on stagnation
            if no_improve > 30:
                F = 0.5
                CR = 1.0
            elif no_improve > 10:
                F = 0.7
                CR = 0.9
            else:
                F = 0.8
                CR = 0.9

            improved = False
            pop_size_current = pop_size

            # Generate offspring for each member
            for i in range(pop_size_current):
                if calls_used >= budget:
                    break

                # Select three distinct random indices different from i
                # Use choice for efficiency
                indices = rng.choice(pop_size_current, size=3, replace=False)
                # Ensure none are i
                while i in indices:
                    indices = rng.choice(pop_size_current, size=3, replace=False)
                a, b, c = indices

                # Choose mutation strategy
                # If stagnating, use DE/best/1 to converge faster
                if no_improve > 20 and i == 0:
                    # For the best individual, try DE/best/1
                    v = pop[0] + F * (pop[a] - pop[b])
                else:
                    # Standard DE/rand/1
                    v = pop[a] + F * (pop[b] - pop[c])

                # Crossover
                j_rand = rng.integers(0, dim)
                trial = np.empty(dim)
                # Vectorized crossover for speed
                mask = rng.random(dim) < CR
                mask[j_rand] = True
                trial[mask] = v[mask]
                trial[~mask] = pop[i][~mask]

                # Clip to bounds
                trial = np.clip(trial, -5, 5)

                # Evaluate
                f_trial = func(trial)
                calls_used += 1

                # Selection
                if f_trial <= fvals[i]:
                    pop[i] = trial
                    fvals[i] = f_trial
                    improved = True

                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

            if improved:
                no_improve = 0
            else:
                no_improve += 1

            # If stuck for too long, inject diversity
            if no_improve > 15:
                # Replace worst 30% with new points
                n_replace = max(1, pop_size_current // 3)
                for i in range(pop_size_current - n_replace, pop_size_current):
                    if calls_used >= budget:
                        break
                    # Mix of local exploration around best and global random
                    if rng.random() < 0.5:
                        # Local exploration: perturb best
                        scale = 0.5 * (1.0 + no_improve / 20.0)
                        pop[i] = best_x + rng.normal(0, scale, dim)
                    else:
                        # Global exploration
                        pop[i] = rng.uniform(-5, 5, dim)
                    
                    pop[i] = np.clip(pop[i], -5, 5)
                    fvals[i] = func(pop[i])
                    calls_used += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = pop[i].copy()
                no_improve = 0

            gen += 1

            # If budget is running low, do a final local polish
            if calls_used > budget * 0.9 and no_improve > 5:
                # Simple local search around best_x
                # Try small perturbations
                for _ in range(5):
                    if calls_used >= budget:
                        break
                    trial = best_x + rng.normal(0, 0.1, dim)
                    trial = np.clip(trial, -5, 5)
                    f_trial = func(trial)
                    calls_used += 1
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
