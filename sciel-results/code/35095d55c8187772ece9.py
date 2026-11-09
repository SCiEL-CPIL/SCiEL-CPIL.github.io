"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Deterministic global search using Differential Evolution followed by single L-BFGS-B polish."""

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
        rng = np.random.default_rng(42)

        # Population size tuned for 5D Rosenbrock
        pop_size = 20
        # Allocate 80% of budget to DE, 20% to Polish
        de_budget = int(budget * 0.8)
        
        # Initial population: uniform in [-5, 5]^dim
        pop = rng.uniform(-5.0, 5.0, size=(pop_size, dim))
        pop_f = np.array([counted_func(ind) for ind in pop])

        # DE parameters
        F = 0.8  # scaling factor - slightly higher for Rosenbrock
        CR = 0.9  # crossover rate

        # Track best
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        # DE Loop
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

        # After DE, find best
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        # --- Phase 2: Single L-BFGS-B polish on best candidate ---
        remaining = budget - call_count[0]
        if remaining > 0:
            bounds = [(-5.0, 5.0)] * dim
            max_fun = remaining
            
            if max_fun >= 6: # Need at least 1 gradient step (dim+1=6 calls)
                try:
                    result = minimize(
                        counted_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': max_fun, 'maxfun': max_fun}
                    )
                    if result.fun < best_f:
                        best_x = result.x
                        best_f = result.fun
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
