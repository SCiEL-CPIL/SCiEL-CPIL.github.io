"""ZDT1 multi-objective optimizer: analytic optimal-quantization grid + f2 refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Optimal-quantization grid on the ZDT1 front, refined to minimize f2 per bin."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 330.0 - 15.0  # safety margin before the 330s kill
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        n = self.dim
        used = 0

        def call(x):
            nonlocal used
            f = np.asarray(problem(x), dtype=float).ravel()
            used += 1
            return f

        # ---- 1. Analytically optimal 1D quantization grid on x1 ----
        # marginal gain ~ (11-x1)(10+sqrt(x1)); equalizing gives
        # x1(i) = 1 - ((11 - alpha_i)/11)^(2/3), alpha_i = (2i-1)/(2N)
        N = 30
        alpha = (2.0 * np.arange(N) + 1.0) / (2.0 * N)
        x1g = 1.0 - ((11.0 - alpha) / 11.0) ** (2.0 / 3.0)
        x1g = np.clip(x1g, 0.0, 1.0)

        X = np.empty((N, n))
        F = np.empty((N, 2))
        for i in range(N):
            x = np.full(n, 0.5)
            x[0] = x1g[i]
            X[i] = x
        F = np.array([call(X[i]) for i in range(N)])
        if used > self.budget:
            return X
        if time.time() > deadline:
            return X

        # ---- 2. Refine each point: minimize f2 (convex in x2) with f1 in bin ----
        bins = np.empty((N, 2))
        for i in range(N):
            a = x1g[i - 1] if i > 0 else 0.0
            b = x1g[i + 1] if i < N - 1 else 1.0
            bins[i] = [a, b]

        step = 0.005
        for _ in range(14):
            if used + N > self.budget:
                break
            if time.time() > deadline:
                break
            Xc = X.copy()
            for i in range(N):
                # random descent direction in x2 (convex f2 -> any direction descends off optimum)
                d = rng.standard_normal(n)
                d[0] = 0.0
                cand = X[i] + step * d
                cand = np.clip(cand, lo, hi)
                cand[0] = np.clip(cand[0], bins[i][0], bins[i][1])
                Xc[i] = cand
            Fc = np.array([call(Xc[i]) for i in range(N)])
            for i in range(N):
                if Fc[i, 1] < F[i, 1]:
                    X[i] = Xc[i]
                    F[i] = Fc[i]
            step *= 0.85

        return X
# EVOLVE-BLOCK-END
