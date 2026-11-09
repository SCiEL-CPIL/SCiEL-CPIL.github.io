"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """DE-based multi-objective search with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Initial population: mix of corners and random points
        n_corners = min(2 ** dim, 32)
        n_random = max(1, 40 - n_corners)

        corners = np.array(np.meshgrid(*[np.array([lo[i], hi[i]]) for i in range(dim)])).T.reshape(-1, dim)
        if n_corners < len(corners):
            idx = rng.choice(len(corners), n_corners, replace=False)
            corners = corners[idx]

        random_pts = lo + (hi - lo) * rng.random((n_random, dim))

        X = np.vstack([corners, random_pts])
        F = np.array([problem(x) for x in X])
        used = len(X)

        def _non_dominated_mask(F_arr):
            """Return boolean mask of non-dominated points."""
            n = len(F_arr)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                diff = F_arr - F_arr[i]
                dominates = np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)
                dominates[i] = False
                if np.any(dominates):
                    mask[i] = False
            return mask

        def _dominance_count(F_arr):
            """Return dominance count for each point."""
            n = len(F_arr)
            counts = np.zeros(n, dtype=int)
            for i in range(n):
                diff = F_arr - F_arr[i]
                dominates = np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)
                dominates[i] = False
                counts[i] = np.sum(dominates)
            return counts

        # DE parameters
        CR = 0.7  # crossover rate
        F_DE = 0.8  # differential weight

        while used < self.budget:
            if time.time() > deadline:
                break

            # Find non-dominated set
            nd_mask = _non_dominated_mask(F)
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) < 3:
                # Not enough non-dominated points for DE, use random parents
                parent_indices = rng.choice(len(F), 3, replace=False)
            else:
                parent_indices = rng.choice(nd_indices, 3, replace=False)

            # Differential Evolution: v = x_r1 + F*(x_r2 - x_r3)
            r1, r2, r3 = parent_indices
            v = X[r1] + F_DE * (X[r2] - X[r3])

            # Crossover: binomial crossover
            j_rand = rng.integers(dim)
            trial = np.empty(dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = X[r1][j]

            # Clip to bounds
            trial = np.clip(trial, lo, hi)

            fc = problem(trial)
            used += 1

            # Selection: keep the better of trial and X[r1] based on domination
            # If trial dominates r1, replace; if r1 dominates trial, keep; if non-dominated, use crowding
            if np.all(fc <= F[r1]) and np.any(fc < F[r1]):
                # trial dominates r1
                X[r1] = trial
                F[r1] = fc
            elif np.all(F[r1] <= fc) and np.any(F[r1] < fc):
                # r1 dominates trial, keep r1
                pass
            else:
                # Non-dominated: use simple heuristic - replace if trial has better sum
                if np.sum(fc) < np.sum(F[r1]):
                    X[r1] = trial
                    F[r1] = fc
                else:
                    # Keep r1, but maybe store trial if we have space
                    # For simplicity, just keep r1
                    pass

        return X
# EVOLVE-BLOCK-END
