"""Differential Evolution optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with small population and best-reference mutation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        lb = -5.0
        ub = 5.0

        # Population parameters
        lam = 5  # population size
        F = 0.8   # mutation scale factor
        CR = 0.7  # crossover rate

        def time_ok():
            return time.time() < deadline and calls_made < budget - 1

        # Initialize population
        rng = np.random.default_rng(42)
        pop = rng.uniform(-4.0, 4.0, (lam, dim))
        pop_f = np.empty(lam)
        for i in range(lam):
            if not time_ok():
                break
            pop_f[i] = func(pop[i])
            calls_made += 1

        # Track best
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        # Main DE loop
        while time_ok():
            # Sort population by fitness
            order = np.argsort(pop_f)
            pop = pop[order]
            pop_f = pop_f[order]

            # Update best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = pop[0].copy()

            # Generate offspring
            for i in range(lam):
                if not time_ok():
                    break

                # Select three distinct random individuals different from i
                candidates = [j for j in range(lam) if j != i]
                j, k, l = rng.choice(candidates, size=3, replace=False)

                # Mutation: best/2/bin - use best individual as reference
                # v = best + F * (pop[j] - pop[k])
                v = pop[0] + F * (pop[j] - pop[k])

                # Binomial crossover
                trial = pop[i].copy()
                jrand = rng.integers(dim)
                for d in range(dim):
                    if rng.random() < CR or d == jrand:
                        trial[d] = v[d]

                # Clip to bounds
                trial = np.clip(trial, lb, ub)

                # Evaluate
                trial_f = func(trial)
                calls_made += 1

                # Selection
                if trial_f <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f

                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial.copy()

        # Final polish: coordinate descent around best
        if best_x is not None and time_ok():
            x = best_x.copy()
            fx = best_f
            step = 0.1

            for _ in range(min(50, budget - calls_made)):
                if not time_ok():
                    break

                improved = False
                for d in range(dim):
                    if not time_ok():
                        break
                    for direction in [1, -1]:
                        y = x.copy()
                        y[d] += direction * step
                        y = np.clip(y, lb, ub)
                        fy = func(y)
                        calls_made += 1
                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
