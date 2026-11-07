"""Decomposition-based optimizer for WFG7 using Tchebycheff scalarization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Decomposition-based multi-objective optimizer using Tchebycheff scalarization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Generate 30 uniformly distributed weight vectors in 3D
        n_weights = 30
        weights = self._generate_weight_vectors(n_weights, self.n_obj, rng)

        # Estimate ideal point (lower bound of each objective) via initial random samples
        n_init = min(10, self.budget // 10)
        X_init = lo + (hi - lo) * rng.random((n_init, self.dim))
        F_init = np.array([problem(x) for x in X_init])
        ideal = np.min(F_init, axis=0)
        # Add small epsilon to avoid division by zero
        ideal = ideal - 1e-6
        used = n_init

        # Reference point for normalization (use upper bounds of objectives from init)
        ref = np.max(F_init, axis=0) + 1.0

        # Store all evaluated points
        all_X = [X_init[i] for i in range(n_init)]
        all_F = [F_init[i] for i in range(n_init)]

        # For each weight vector, run a local search (Nelder-Mead) on the Tchebycheff scalarized objective
        # We have budget - n_init evaluations left
        remaining = self.budget - n_init
        evals_per_sub = max(1, remaining // n_weights)

        for w in weights:
            # Define the Tchebycheff scalarized objective
            def tcheb_obj(x, w=w, ideal=ideal, ref=ref):
                x = np.clip(x, lo, hi)
                f = np.array(problem(x))
                # Normalize objectives
                f_norm = (f - ideal) / (ref - ideal + 1e-10)
                # Tchebycheff scalarization: max_i { w_i * |f_i - z_i^i| }
                # Use the standard form: max_i { w_i * f_i_norm }
                return np.max(w * f_norm)

            # Initialize at center of search space with some randomness
            x0 = 0.5 * (lo + hi) + 0.1 * (hi - lo) * rng.standard_normal(self.dim)
            x0 = np.clip(x0, lo, hi)

            # Count evaluations
            count = [0]
            def wrapped_obj(x):
                count[0] += 1
                return tcheb_obj(x)

            # Run Nelder-Mead with limited function evaluations
            options = {
                'maxiter': 50,
                'maxfev': evals_per_sub,
                'xatol': 1e-6,
                'fatol': 1e-6
            }

            try:
                result = minimize(wrapped_obj, x0, method='Nelder-Mead', options=options)
                if result.fun < np.inf:
                    x_best = np.clip(result.x, lo, hi)
                    f_best = np.array(problem(x_best))
                    all_X.append(x_best)
                    all_F.append(f_best)
                    used += count[0]
            except Exception:
                pass

            # If we still have budget, do a few more random perturbations around the best found
            if used < self.budget:
                extra = min(3, self.budget - used)
                for _ in range(extra):
                    if used >= self.budget:
                        break
                    x_pert = x_best + 0.05 * (hi - lo) * rng.standard_normal(self.dim)
                    x_pert = np.clip(x_pert, lo, hi)
                    f_pert = np.array(problem(x_pert))
                    all_X.append(x_pert)
                    all_F.append(f_pert)
                    used += 1

        # Final: ensure we haven't exceeded budget
        # Return the last point (return value is ignored, but must not raise)
        return all_X[-1] if all_X else lo.copy()

    def _generate_weight_vectors(self, n, n_obj, rng):
        """Generate n uniformly distributed weight vectors in n_obj dimensions summing to 1."""
        weights = []
        if n_obj == 1:
            return [np.array([1.0])]
        
        # Use Dirichlet distribution with alpha=1 for uniform distribution on simplex
        # But for better coverage, use a grid-like approach for 3D
        if n_obj == 3:
            # Generate points on the 2D simplex
            # Use stratified sampling
            n_side = int(np.ceil(np.sqrt(n)))
            weights = []
            for i in range(n_side):
                for j in range(n_side - i):
                    k = n_side - i - j - 1
                    if k < 0:
                        continue
                    w = np.array([i, j, k], dtype=float)
                    s = w.sum()
                    if s > 0:
                        w = w / s
                    else:
                        w = np.array([1.0/3, 1.0/3, 1.0/3])
                    weights.append(w)
                    if len(weights) >= n:
                        break
                if len(weights) >= n:
                    break
            # Pad if needed
            while len(weights) < n:
                w = rng.dirichlet([1, 1, 1])
                weights.append(w)
            return weights[:n]
        else:
            # General case: use Dirichlet
            weights = [rng.dirichlet([1] * n_obj) for _ in range(n)]
            return weights


# EVOLVE-BLOCK-END
