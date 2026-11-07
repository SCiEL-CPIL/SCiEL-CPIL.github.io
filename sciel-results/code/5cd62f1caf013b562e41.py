"""Population-based Differential Evolution with adaptive mutation and Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """DE/rand/1 with adaptive F, population size 12, plus Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        pop_size = 12
        # Budget allocation: DE phase uses most, polish uses ~20 calls
        polish_budget = 20
        de_budget = max(budget - polish_budget, 10)

        # Initialize population uniformly in [-5, 5]^dim
        pop = rng.uniform(-5, 5, size=(pop_size, dim))
        fits = np.array([func(ind) for ind in pop])

        # Track best
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = fits[best_idx]

        # Adaptive mutation factor F: start at 0.8, adapt based on success rate
        F = 0.8
        success_count = 0
        trial_count = 0

        calls_used = pop_size  # initial population evaluations

        # DE loop
        while calls_used < de_budget:
            # Adaptive F: increase if recent success rate is high, decrease if low
            if trial_count > 0:
                success_rate = success_count / trial_count
                if success_rate > 0.5:
                    F = min(1.0, F + 0.05)
                elif success_rate < 0.2:
                    F = max(0.3, F - 0.05)
                # Reset counters every so often
                if trial_count >= 20:
                    success_count = 0
                    trial_count = 0

            for i in range(pop_size):
                if calls_used >= de_budget:
                    break

                # Select three distinct random indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # DE/rand/1 mutation
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, -5, 5)

                # Evaluate mutant
                fm = func(mutant)
                calls_used += 1
                trial_count += 1

                # Selection: replace if mutant is better
                if fm < fits[i]:
                    pop[i] = mutant
                    fits[i] = fm
                    success_count += 1

                    # Update global best
                    if fm < best_f:
                        best_f = fm
                        best_x = mutant.copy()

        # Nelder-Mead polish on the best solution found by DE
        # Use a simple bounded Nelder-Mead via scipy's minimize with bounds
        if calls_used < budget:
            # Set up Nelder-Mead with a few iterations
            # We'll use a wrapper that clips to bounds
            def bounded_func(x):
                x_clipped = np.clip(x, -5, 5)
                return func(x_clipped)

            # Initial simplex around best_x
            initial_simplex = [best_x]
            for d in range(dim):
                perturbation = best_x.copy()
                step = 0.5 if best_x[d] < 4.5 else -0.5
                perturbation[d] += step
                perturbation = np.clip(perturbation, -5, 5)
                initial_simplex.append(perturbation)

            # Count how many more function evaluations we can afford
            # Nelder-Mead with dim=5 uses about dim+1=6 points initially
            # Each iteration uses ~1-2 evaluations
            nm_budget = budget - calls_used
            maxiter = max(1, nm_budget // 3)  # conservative estimate

            try:
                result = minimize(
                    bounded_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
