"""Analytical multi-objective optimizer for ZDT4."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit ZDT4 structure: g is minimized when each x_i is near +/-1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _nondominated(self, F):
        n = len(F)
        nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd[i]:
                continue
            for j in range(n):
                if i == j or not nd[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] = False
        return nd

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280.0  # safety margin before 300s limit
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)
        dim = self.dim

        # Target: each coordinate near +/-1 to minimize g; vary magnitude for f1 spread.
        X = []
        F = []
        used = 0

        def add(x):
            nonlocal used
            fc = problem(x)
            X.append(x)
            F.append(fc)
            used += 1

        # Phase 1: sample sign patterns with varied magnitudes to spread f1.
        # For each coordinate pick a sign and a magnitude in a range.
        n_init = min(120, self.budget - 20)
        for k in range(n_init):
            signs = np.where(rng.random(dim) < 0.5, 1.0, -1.0)
            # magnitudes: mix of near 1 (low g) and some spread for f1
            mags = np.ones(dim)
            # pick a few coords to have larger magnitude to increase f1
            n_big = rng.integers(0, 5)
            if n_big > 0:
                idx = rng.choice(dim, size=n_big, replace=False)
                mags[idx] = rng.uniform(1.0, 3.0, size=n_big)
            x = signs * mags
            x = np.clip(x, lo, hi)
            add(x)

        # Phase 2: local refinement around best (lowest g) points, small steps.
        F = np.array(F)
        # find points with low f2 (proxy for low g)
        order = np.argsort(F[:, 1])
        for k in order[:30]:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            base = X[k]
            for _ in range(3):
                if used >= self.budget or time.time() > deadline:
                    break
                x = base + 0.05 * (hi - lo) * rng.standard_normal(dim)
                x = np.clip(x, lo, hi)
                add(x)

        # Phase 3: fill remaining budget with random exploration near +/-1.
        while used < self.budget and time.time() < deadline:
            signs = np.where(rng.random(dim) < 0.5, 1.0, -1.0)
            mags = rng.uniform(0.5, 2.5, size=dim)
            x = signs * mags
            x = np.clip(x, lo, hi)
            add(x)

        F = np.array(F)
        return X
# EVOLVE-BLOCK-END
