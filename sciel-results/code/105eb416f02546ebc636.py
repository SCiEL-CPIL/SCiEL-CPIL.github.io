"""Decomposition-based multi-objective optimizer for WFG7 using Tchebycheff scalarization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Decomposition-based multi-objective optimizer using Tchebycheff scalarization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s wall-clock limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        center = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0

        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # Generate weight vectors for Tchebycheff scalarization
        # For 3 objectives, use a simplex grid
        n_weights = 30
        weights = self._generate_weights(n_weights, n_obj)

        # Reference point for scalarization (use midpoint of bounds in objective space)
        # We'll estimate it from initial evaluations
        ref_point = None

        # Initial population: evaluate center and a few random points
        rng = np.random.default_rng(42)

        # Evaluate center first
        x_center = center.copy()
        f_center = np.array(problem(x_center))
        ref_point = f_center.copy()

        used = 1

        # Generate initial population
        pop_size = min(10, budget - used)
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))

        # First point is center
        X[0] = x_center
        F[0] = f_center

        for i in range(1, pop_size):
            x = lo + (hi - lo) * rng.random(dim)
            X[i] = x
            F[i] = np.array(problem(x))
            used += 1
            if used >= budget:
                break

        # Update reference point to be the max of all objectives seen
        ref_point = np.max(F, axis=0)

        # Best solutions found so far
        best_solutions = [(X[i].copy(), F[i].copy()) for i in range(len(X))]

        # For each weight vector, solve a single-objective subproblem
        evals_per_subproblem = max(1, (budget - used) // n_weights)

        for w in weights:
            if used >= budget:
                break
            elapsed = time.time() - start_time
            if elapsed > deadline - 30:  # 30s safety margin
                break

            # Tchebycheff scalarization: min max_i (w_i * (f_i(x) - z_i))
            # where z_i is the ideal point (we approximate with ref_point)

            def scalarized_objective(x, w=w):
                if time.time() - start_time > deadline - 10:
                    return 1e10
                f = np.array(problem(x))
                used[0] += 1
                return np.max(w * (f - ref_point))

            # Use a mutable counter for tracking evaluations
            used_counter = [0]

            # Multiple random restarts for each subproblem
            n_restarts = 3
            best_scalar = np.inf
            best_x = None

            for restart in range(n_restarts):
                if used + used_counter[0] >= budget:
                    break
                elapsed = time.time() - start_time
                if elapsed > deadline - 30:
                    break

                # Initialize at random point
                x0 = lo + (hi - lo) * rng.random(dim)

                try:
                    result = minimize(
                        scalarized_objective,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': max(5, evals_per_subproblem // 3),
                            'xatol': 1e-6,
                            'fatol': 1e-6
                        }
                    )
                    if result.fun < best_scalar:
                        best_scalar = result.fun
                        best_x = result.x
                except Exception:
                    pass

                if used + used_counter[0] >= budget:
                    break

            if best_x is not None:
                f_best = np.array(problem(best_x))
                used += 1
                best_solutions.append((best_x.copy(), f_best.copy()))

        # Return the set of best solutions
        return best_solutions

    def _generate_weights(self, n_weights, n_obj):
        """Generate uniformly distributed weight vectors on the simplex."""
        if n_obj == 3:
            # Use a 2D grid on the simplex for 3 objectives
            n_per_axis = int(np.ceil(np.cbrt(n_weights)))
            weights = []
            for i in range(n_per_axis + 1):
                for j in range(n_per_axis + 1 - i):
                    k = n_per_axis - i - j
                    w = np.array([i, j, k], dtype=float)
                    w_sum = w.sum()
                    if w_sum > 0:
                        w = w / w_sum
                    else:
                        w = np.ones(3) / 3.0
                    weights.append(w)
                    if len(weights) >= n_weights:
                        break
                if len(weights) >= n_weights:
                    break

            # Pad if needed
            while len(weights) < n_weights:
                w = np.random.dirichlet(np.ones(n_obj))
                weights.append(w)

            return weights[:n_weights]
        else:
            # General case: random Dirichlet
            rng = np.random.default_rng(0)
            return [rng.dirichlet(np.ones(n_obj)) for _ in range(n_weights)]
# EVOLVE-BLOCK-END
