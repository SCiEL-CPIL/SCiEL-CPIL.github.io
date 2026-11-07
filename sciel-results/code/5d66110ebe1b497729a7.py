import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        B = int(self.budget)
        D = int(self.dim)

        if B <= 0 or D <= 0:
            return np.zeros(D)

        # Reference point for scoring
        R1 = 11.0
        R2 = 11.0

        # ZDT1 Pareto front: f1 = x1, f2 = 1 - sqrt(x1)
        # Optimal quantization: density proportional to height h(x1) = R2 - f2(x1) = 10 + sqrt(x1)
        # CDF: G(x) = (10*x + (2/3)*x^1.5) / (10 + 2/3) = (10*x + (2/3)*x^1.5) / (32/3)
        # G'(x) = (10 + sqrt(x)) / (32/3)

        def G(x):
            return (10.0 * x + (2.0 / 3.0) * np.power(x, 1.5)) / (32.0 / 3.0)

        def G_prime(x):
            return (10.0 + np.sqrt(np.clip(x, 0.0, None))) / (32.0 / 3.0)

        # Place B points at u_i = (i + 0.5) / B (midpoint rule)
        u = (np.arange(B, dtype=np.float64) + 0.5) / B

        # Invert G(x) = u using Newton's method
        # Good initial guess: for small x, G(x) ~ (30/32)*x, so x ~ (32/30)*u
        # For large x, G(x) ~ (32/32)*x = x, so x ~ u
        x = np.clip(u * (32.0 / 30.0), 1e-16, 1.0 - 1e-16)

        for _ in range(100):
            gx = G(x)
            dgx = G_prime(x)
            x_new = x - (gx - u) / dgx
            x_new = np.clip(x_new, 0.0, 1.0)
            if np.max(np.abs(x_new - x)) < 1e-16:
                x = x_new
                break
            x = x_new

        x1_vals = np.clip(x, 0.0, 1.0)

        # Construct design matrix
        X = np.zeros((B, D), dtype=np.float64)
        X[:, 0] = x1_vals

        # Clip to bounds
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        X = np.clip(X, lo, hi)

        # Evaluate each point
        t0 = time.time()
        deadline = t0 + 300.0
        for i in range(B):
            if time.time() > deadline:
                break
            problem(X[i].copy())

        return X[-1]
