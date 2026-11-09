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

        # Population size: balance exploration vs exploitation
        pop_size = 24
        # Allocate 80% of budget to DE, 20% to Polish
        de_budget = int(budget * 0.8)
        
        # Initial population: stratified sampling for better coverage
        # Use Latin Hypercube-style initialization
        pop = np.zeros((pop_size, dim))
        for j in range(dim):
            # Stratified sampling along each dimension
            edges = np.linspace(-5.0, 5.0, pop_size + 1)
            midpoints = (edges[:-1] + edges[1:]) / 2.0
            # Shuffle to decorrelate dimensions
            perm = rng.permutation(pop_size)
            pop[:, j] = midpoints[perm]

        pop_f = np.array([counted_func(ind) for ind in pop])

        # DE parameters with adaptive F
        CR = 0.85  # crossover rate

        # Track best
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        # Adaptive F: start high for exploration, decrease over time
        F_start = 0.8
        F_end = 0.3
        total_de_evals = de_budget - pop_size
        de_evals_done = 0

        # DE Loop
        while call_count[0] < de_budget:
            # Find current best
            best_idx = np.argmin(pop_f)
            best_x = pop[best_idx].copy()
            best_f = pop_f[best_idx]

            # Adaptive F based on progress
            progress = min(1.0, de_evals_done / max(1, total_de_evals))
            F = F_start + (F_end - F_start) * progress

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
                de_evals_done += 1

                # Selection
                if trial_f <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f

        # After DE, find best
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        # --- Phase 2: Single high-precision L-BFGS-B polish ---
        remaining = budget - call_count[0]
        if remaining >= 6:  # Need at least 1 gradient step (dim+1=6 calls)
            bounds = [(-5.0, 5.0)] * dim
            max_fun = remaining
            
            try:
                result = minimize(
                    counted_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_fun, 'maxfun': max_fun, 'gtol': 1e-10, 'ftol': 1e-15}
                )
                if result.fun < best_f:
                    best_x = result.x
                    best_f = result.fun
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
