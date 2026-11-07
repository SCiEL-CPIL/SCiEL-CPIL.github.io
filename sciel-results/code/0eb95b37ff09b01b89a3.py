"""Population-based Differential Evolution with adaptive mutation and Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """DE/rand/1 with adaptive F, population size 15, plus multi-restart Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size: balance exploration and exploitation
        pop_size = 15
        # Budget allocation: DE phase uses most, polish uses ~30 calls
        polish_budget = 30
        de_budget = max(budget - polish_budget, 10)

        # Initialize population: LHS-inspired spread for better coverage
        pop = np.empty((pop_size, dim))
        for i in range(pop_size):
            for d in range(dim):
                # Use stratified sampling: divide [-5,5] into pop_size bins
                bin_idx = (i * dim + d) % pop_size
                lower = -5.0 + 10.0 * bin_idx / pop_size
                upper = -5.0 + 10.0 * (bin_idx + 1) / pop_size
                pop[i, d] = rng.uniform(lower, upper)
        
        # Add a few Gaussian samples around center for local exploitation
        n_gauss = min(3, pop_size)
        for i in range(n_gauss):
            pop[i] = rng.normal(0, 1.5, size=dim)
        
        pop = np.clip(pop, -5, 5)

        fits = np.array([func(ind) for ind in pop])

        # Track best
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = fits[best_idx]

        # Adaptive mutation factor F: start at 0.8 for more exploration initially
        F = 0.8
        success_count = 0
        trial_count = 0
        last_best_f = best_f
        no_improve_gen = 0

        calls_used = pop_size  # initial population evaluations

        # DE loop
        while calls_used < de_budget:
            # Adaptive F: faster adaptation every 8 trials
            if trial_count >= 8:
                success_rate = success_count / trial_count
                # Target F: higher when success rate is low (need more exploration)
                target_F = 0.5 + 0.6 * (1.0 - success_rate)
                F = 0.7 * F + 0.3 * np.clip(target_F, 0.3, 1.2)
                success_count = 0
                trial_count = 0

            # Check for stagnation
            if best_f < last_best_f - 1e-10:
                no_improve_gen = 0
                last_best_f = best_f
            else:
                no_improve_gen += 1

            # If stuck for many generations, inject diversity more aggressively
            if no_improve_gen > 4:
                worst_indices = np.argsort(fits)[-4:]
                for idx in worst_indices:
                    # Mix of uniform and Gaussian restarts
                    if rng.random() < 0.5:
                        pop[idx] = rng.uniform(-5, 5, size=dim)
                    else:
                        pop[idx] = rng.normal(0, 2.0, size=dim)
                    pop[idx] = np.clip(pop[idx], -5, 5)
                    fits[idx] = func(pop[idx])
                    calls_used += 1
                    if fits[idx] < best_f:
                        best_f = fits[idx]
                        best_x = pop[idx].copy()
                    if calls_used >= de_budget:
                        break

            for i in range(pop_size):
                if calls_used >= de_budget:
                    break

                # Select three distinct random indices different from i
                candidates = np.array([j for j in range(pop_size) if j != i])
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

        # Nelder-Mead polish: single focused polish on best solution
        if calls_used < budget:
            call_counter = {'count': calls_used}

            def bounded_func(x):
                x_clipped = np.clip(x, -5, 5)
                call_counter['count'] += 1
                return func(x_clipped)

            remaining = budget - calls_used
            if remaining > 5:
                try:
                    result = minimize(
                        bounded_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': max(1, remaining // 3),
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True,
                            'maxfev': remaining
                        }
                    )
                    calls_used = call_counter['count']
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5, 5)
                except Exception:
                    calls_used = call_counter['count']

        return best_x, best_f
# EVOLVE-BLOCK-END
