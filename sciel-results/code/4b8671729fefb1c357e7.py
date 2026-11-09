"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Deterministic global search using Differential Evolution followed by L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Count function calls
        call_count = [0]

        def counted_func(x):
            x = np.asarray(x, dtype=float).ravel()
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            call_count[0] += 1
            return val

        # --- Phase 1: Differential Evolution (global search) ---
        # Use a fixed seed for determinism
        rng = np.random.default_rng(42)

        # Population size
        pop_size = 20
        # Reserve budget for L-BFGS-B polish
        lbfgs_budget = 150
        de_budget = budget - lbfgs_budget

        # Initial population: uniform in [-5, 5]^dim
        pop = rng.uniform(-5.0, 5.0, size=(pop_size, dim))
        pop_f = np.array([counted_func(ind) for ind in pop])

        # DE parameters
        F = 0.8  # scaling factor
        CR = 0.9  # crossover rate

        # Track best
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        # Number of DE generations we can afford
        # Each generation uses pop_size function evaluations (for mutation/crossover trials)
        # But we evaluate trial vectors one by one to track budget
        gen = 0
        while call_count[0] < de_budget:
            # Find current best
            best_idx = np.argmin(pop_f)
            best_x = pop[best_idx].copy()
            best_f = pop_f[best_idx]

            for i in range(pop_size):
                if call_count[0] >= de_budget:
                    break

                # Select three distinct random indices different from i
                mask = np.ones(pop_size, dtype=bool)
                mask[i] = False
                candidates = np.where(mask)[0]
                if len(candidates) < 3:
                    break
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # Mutation: v = pop[a] + F * (pop[b] - pop[c])
                v = pop[a] + F * (pop[b] - pop[c])

                # Crossover
                j_rand = rng.integers(0, dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]

                # Clip to bounds
                trial = np.clip(trial, -5.0, 5.0)

                # Evaluate trial
                trial_f = counted_func(trial)

                # Selection
                if trial_f <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f

            gen += 1

        # After DE, find best
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        # --- Phase 2: L-BFGS-B polish ---
        # Use the best candidate from DE as starting point
        # We'll do a single L-BFGS-B run with maxiter limited by remaining budget
        remaining = budget - call_count[0]
        if remaining > 0:
            try:
                # L-BFGS-B with bounds
                bounds = [(-5.0, 5.0)] * dim
                result = minimize(
                    counted_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': remaining, 'maxfun': remaining}
                )
                if result.fun < best_f:
                    best_x = result.x
                    best_f = result.fun
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
