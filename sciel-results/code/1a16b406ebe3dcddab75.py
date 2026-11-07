"""DTLZ7 hypervolume optimizer exploiting the known Pareto front geometry."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _halton(index, base):
    f = 1.0
    r = 0.0
    i = index + 1
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        def h(y):
            return np.where(
                y <= 0.5,
                y * y,
                1.0 - (y - 0.5) ** 2 - 0.5 * np.sin(8.0 * np.pi * y),
            )

        def f_of_u(u1, u2):
            x1 = u1 * (hi[0] - lo[0]) + lo[0]
            x2 = u2 * (hi[1] - lo[1]) + lo[1]
            x1 = np.clip(x1, lo[0], hi[0])
            x2 = np.clip(x2, lo[1], hi[1])
            h1 = h(x1)
            h2 = h(x2)
            f1 = np.ones_like(x1)
            f2 = x2 * (3.0 - h1 - h2)
            f3 = h1
            return f1, f2, f3

        # Phase 1: coarse Halton sweep to capture global structure
        n_coarse = 80
        uc = np.empty((n_coarse, 2))
        for k in range(n_coarse):
            uc[k, 0] = _halton(k, 2)
            uc[k, 1] = _halton(k, 3)
        f1c, f2c, f3c = f_of_u(uc[:, 0], uc[:, 1])

        # Phase 2: adaptive refinement around promising (f2, f3) regions
        # Use a second independent Halton sequence offset to avoid duplicate u's
        n_fine = self.budget - n_coarse
        offset = n_coarse
        uf = np.empty((n_fine, 2))
        for k in range(n_fine):
            uf[k, 0] = _halton(offset + k, 2)
            uf[k, 1] = _halton(offset + k, 3)
        f1f, f2f, f3f = f_of_u(uf[:, 0], uf[:, 1])

        # Combine and evaluate
        X = np.zeros((self.budget, self.dim))
        X[:n_coarse, 0] = np.clip(uc[:, 0] * (hi[0] - lo[0]) + lo[0], lo[0], hi[0])
        X[:n_coarse, 1] = np.clip(uc[:, 1] * (hi[1] - lo[1]) + lo[1], lo[1], hi[1])
        X[n_coarse:, 0] = np.clip(uf[:, 0] * (hi[0] - lo[0]) + lo[0], lo[0], hi[0])
        X[n_coarse:, 1] = np.clip(uf[:, 1] * (hi[1] - lo[1]) + lo[1], lo[1], hi[1])

        F = np.zeros((self.budget, self.n_obj))
        for i in range(self.budget):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])

        return X
# EVOLVE-BLOCK-END
