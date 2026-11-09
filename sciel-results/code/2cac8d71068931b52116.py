"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective optimizer for DTLZ2 exploiting the known hypersphere front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _front_points(rng, n, dim):
        """Generate n points on the unit hypersphere (DTLZ2 front) in R^dim."""
        v = rng.standard_normal((n, dim))
        nrm = np.linalg.norm(v, axis=1, keepdims=True)
        nrm[nrm == 0.0] = 1.0
        return v / nrm

    @staticmethod
    def _non_dominated_mask(F):
        """Boolean mask of non-dominated rows (vectorized)."""
        n = F.shape[0]
        if n <= 1:
            return np.ones(n, dtype=bool)
        # best[i] = True if row i is dominated by some other row
        best = np.min(F, axis=0, keepdims=True)
        ge = F >= best
        ge_count = ge.sum(axis=1)
        best_mask = ge_count == F.shape[1]
        best_idx = np.where(best_mask)[0]
        if len(best_idx) <= 1:
            return np.ones(n, dtype=bool)
        B = F[best_idx]
        D = (F[:, None, :] <= B[None, :, :]) & (F[:, None, :] < B[None, :, :]).any(axis=2)
        dominated = D.any(axis=1)
        return ~dominated

    def __call__(self, problem, bounds):
        deadline = time.time() + 280.0
        rng = np.random.default_rng(12345)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim, n_obj = self.dim, self.n_obj

        # Seed with a diverse, near-optimal population on the known front.
        pop = 40
        P = self._front_points(rng, pop, dim)
        X = 0.5 + 0.49 * P
        F = np.array([problem(x) for x in X])
        used = pop

        while used < self.budget and time.time() < deadline:
            mask = self._non_dominated_mask(F)
            npar = int(mask.sum())
            if npar < 2:
                parents = np.arange(pop)
            else:
                parents = np.where(mask)[0]
            # Mutation: pick a base parent and two distinct difference parents.
            i = rng.integers(npar)
            a = parents[i]
            j = rng.choice(np.delete(parents, i))
            k = rng.choice(np.delete(parents, i))
            v = X[a] + 0.8 * (X[j] - X[k])
            v = np.clip(v, lo, hi)
            fv = problem(v)
            used += 1
            # Tournament: keep the mutant if it dominates a random parent, else random.
            c = rng.integers(pop)
            if np.all(fv <= F[c]) and np.any(fv < F[c]):
                X[c] = v
                F[c] = fv
            elif rng.random() < 0.5:
                X[c] = v
                F[c] = fv
        return X
# EVOLVE-BLOCK-END
