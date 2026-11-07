"""Multi-objective WFG7 optimizer: Sobol init + Nelder-Mead local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Phase 1: Deterministic Sobol space-filling initialization
        # Use a power-of-2 count for Sobol quality
        n_sobol = 128
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        u = sobol.random(n_sobol)
        X_sobol = lo + u * span
        F_sobol = np.array([problem(x) for x in X_sobol])
        used = n_sobol

        # Phase 2: Find non-dominated points from Sobol sample
        nondom_idx = self._nondominated_indices(F_sobol)
        # Sort non-dominated by sum of objectives (proxy for closeness to ideal)
        sums = np.sum(F_sobol[nondom_idx], axis=1)
        sorted_nd = nondom_idx[np.argsort(sums)]

        # Phase 3: Local refinement via Nelder-Mead on top non-dominated points
        # Allocate remaining budget to local optimization
        remaining = budget - used
        # Each Nelder-Mead call uses ~dim+1 initial simplex + iterations
        # Estimate cost per local opt: ~2*(dim+1) evaluations (conservative)
        est_cost_per_opt = 2 * (dim + 1)
        n_local = min(len(sorted_nd), max(1, remaining // est_cost_per_opt))

        # Select top-n non-dominated points for local refinement
        local_starts = sorted_nd[:n_local]

        # For each start, run Nelder-Mead with a scalarized objective
        # We use weighted sum with different weight vectors to explore different parts of PF
        # Generate weight vectors on the simplex
        n_weights = min(n_local, 10)
        # Create diverse weight vectors
        rng_local = np.random.default_rng(123)
        weights = []
        for _ in range(n_weights):
            w = rng_local.dirichlet(np.ones(n_obj))
            weights.append(w)
        weights = np.array(weights)

        # Also include the original non-dominated points as candidates
        all_candidates_X = X_sobol[nondom_idx].copy()
        all_candidates_F = F_sobol[nondom_idx].copy()

        for i in range(n_local):
            if time.time() > deadline:
                break
            if used >= budget:
                break

            start_idx = local_starts[i]
            x0 = X_sobol[start_idx].copy()
            w = weights[i % n_weights]

            # Define scalarized objective
            def obj(x):
                x_clip = np.clip(x, lo, hi)
                f = problem(x_clip)
                return np.dot(f, w)

            # Nelder-Mead with bounded constraints handled by clipping
            try:
                res = minimize(
                    obj, x0, method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                x_best = np.clip(res.x, lo, hi)
                f_best = problem(x_best)
                used += 1  # The final evaluation

                # Add to candidates
                all_candidates_X = np.vstack([all_candidates_X, x_best.reshape(1, -1)])
                all_candidates_F = np.vstack([all_candidates_F, f_best.reshape(1, -1)])
            except Exception:
                pass

        # Phase 4: Extract final non-dominated set from all candidates
        final_nondom = self._nondominated_indices(all_candidates_F)
        result_X = all_candidates_X[final_nondom]
        result_F = all_candidates_F[final_nondom]

        return result_X

    def _nondominated_indices(self, F):
        """Return indices of non-dominated points in F."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        nondom = np.ones(n, dtype=bool)
        for i in range(n):
            if not nondom[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nondom[i] = False
                    break
        return np.where(nondom)[0]
# EVOLVE-BLOCK-END
