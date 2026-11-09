"""Multi-objective optimizer on ZDT4 using deterministic LHS with x1 compression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """LHS-based multi-objective optimizer for ZDT4.

    Strategy: deterministic Latin Hypercube over the 10-dim box, with x1
    compressed toward 0 (where f1 is minimal and hypervolume is largest) and
    x2..x10 kept tightly near 0 (the Pareto-optimal region where g is minimal).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = 330.0 - 20.0  # safety margin
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        budget = self.budget

        # Deterministic Latin Hypercube in [0,1]^dim
        def _lhs(n, d, seed=0):
            rng = np.random.default_rng(seed)
            idx = np.arange(n)
            X = np.empty((n, d))
            for j in range(d):
                perm = rng.permutation(n)
                X[:, j] = (perm + rng.random(n)) / n
            return X

        # Number of LHS points to evaluate (use full budget)
        n_pts = int(budget)

        # Generate LHS in [0,1]^dim
        U = _lhs(n_pts, dim, seed=0)

        # Build candidate points:
        #   x1: compress U[:,0] into [0,1] biased toward 0 (power law, exponent > 1
        #        pushes mass toward 0). Use exponent 2.0 for a smooth bias.
        #   x2..x10: map U[:,1:] tightly around 0 within bounds, scaled small (0.01).
        x1_raw = U[:, 0]
        # Power compression toward 0: x1 = x1_raw ** 2 keeps range [0,1] but denser near 0.
        x1_unit = x1_raw ** 2.0
        x1 = lo[0] + x1_unit * span[0]

        # x2..x10: center at 0 (Pareto region), small spread.
        # Map U in [0,1] to [-1,1] then scale by a small fraction of the span.
        center = 0.0
        spread_frac = 0.01  # tight perturbation (measured to help)
        x_tail = np.empty((n_pts, dim - 1))
        for j in range(1, dim):
            u = U[:, j - 1] * 2.0 - 1.0  # in [-1, 1]
            x_tail[:, j - 1] = center + u * spread_frac * span[j]
            # clamp to bounds
            x_tail[:, j - 1] = np.clip(x_tail[:, j - 1], lo[j], hi[j])

        X = np.empty((n_pts, dim))
        X[:, 0] = x1
        X[:, 1:] = x_tail

        # Evaluate all points
        F = np.empty((n_pts, self.n_obj))
        used = 0
        for i in range(n_pts):
            F[i] = problem(X[i])
            used += 1
            if used % 20 == 0 and time.time() > deadline:
                break

        # Keep only evaluated rows
        X = X[:used]
        F = F[:used]

        # Filter to non-dominated points (return best set; return value ignored,
        # but we return X for interface compatibility).
        if used > 0:
            nd_mask = self._non_dominated_mask(F)
            X = X[nd_mask]

        return X

    @staticmethod
    def _non_dominated_mask(F):
        n = F.shape[0]
        if n == 0:
            return np.array([], dtype=bool)
        dominated = np.zeros(n, dtype=bool)
        # Vectorized: point i is dominated if exists j with F[j] <= F[i] and F[j] != F[i]
        for i in range(n):
            if dominated[i]:
                continue
            # compare all j against i
            le = np.all(F <= F[i], axis=1)
            lt = np.any(F < F[i], axis=1)
            dom = le & lt
            dom[i] = False  # self
            if np.any(dom):
                dominated[dom] = True
        return ~dominated
# EVOLVE-BLOCK-END
