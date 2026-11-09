"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MOEA/D-DE style decomposition-based multi-objective optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        N = 30
        # Neighborhood size
        T = 10

        # Weight vectors (Tchebycheff)
        weights = np.zeros((N, n_obj))
        for i in range(N):
            # Simple grid-like weights for 3 objectives
            for k in range(n_obj):
                weights[i, k] = (i // (N // n_obj)) / max(1, N // n_obj - 1)
            # Normalize
            w_sum = weights[i].sum()
            if w_sum > 0:
                weights[i] /= w_sum
            else:
                weights[i] = 1.0 / n_obj

        # Initialize population
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N

        # Ideal point
        F_min = F.min(axis=0)

        # Neighborhood indices
        neighbors = np.array([
            list(range(max(0, i - T // 2), min(N, i + T // 2 + 1)))
            for i in range(N)
        ])

        # DE parameters
        CR = 0.5  # crossover rate
        F_DE = 0.8  # differential weight

        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # For each subproblem, generate a child
            i = rng.integers(N)
            # Pick two parents from neighborhood
            nb = neighbors[i]
            p1_idx, p2_idx = rng.choice(nb, size=2, replace=False)

            # DE mutation
            v = X[p1_idx] + F_DE * (X[p2_idx] - X[p1_idx]) + 0.1 * (hi - lo) * rng.standard_normal(dim)
            v = np.clip(v, lo, hi)

            # Crossover
            child = np.where(rng.random(dim) < CR, v, X[i])
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Update using Tchebycheff scalarization
            # Compute Tchebycheff value for child
            F_min_new = np.minimum(F_min, fc)
            t_child = np.max(weights[i] * np.abs(fc - F_min_new) / np.maximum(weights[i], 1e-8))

            # Replace if better or with some probability
            t_current = np.max(weights[i] * np.abs(F[i] - F_min) / np.maximum(weights[i], 1e-8))
            if t_child <= t_current or rng.random() < 0.1:
                X[i] = child
                F[i] = fc
                F_min = np.minimum(F_min, fc)

            # Local replacement: update neighbors if child is better for them
            for j in neighbors[i]:
                if j == i:
                    continue
                t_j_child = np.max(weights[j] * np.abs(fc - F_min) / np.maximum(weights[j], 1e-8))
                t_j_curr = np.max(weights[j] * np.abs(F[j] - F_min) / np.maximum(weights[j], 1e-8))
                if t_j_child < t_j_curr:
                    X[j] = child
                    F[j] = fc

        return X
# EVOLVE-BLOCK-END
