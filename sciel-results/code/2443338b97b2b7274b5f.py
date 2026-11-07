"""CMA-ES optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # CMA-ES parameters
        # lambda: number of offspring per generation
        lam = max(4 * d, 8)
        # mu: number of best offspring used for recombination
        mu = lam // 2
        # weights for recombination
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        # step size adaptation
        c_sigma = (mueff + 2) / (d + mueff + 5)
        # position update
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        # chi_n: expected norm of N(0, I)
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # Initialize
        x_mean = np.zeros(d)
        sigma = 2.0
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        C = np.eye(d)
        C_sqrt = np.eye(d)
        C_inv = np.eye(d)
        invsqrtC = np.eye(d)

        def update_matrices():
            nonlocal C, C_sqrt, C_inv, invsqrtC
            # Eigendecomposition for square root
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            invsqrtC = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        update_matrices()

        best_x = None
        best_f = np.inf
        calls = 0

        def safe_eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            calls += 1
            try:
                f = float(func(x))
            except Exception:
                return None
            if not np.isfinite(f):
                return None
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial evaluation
        fx = safe_eval(x_mean)
        if fx is None:
            return x_mean, np.inf

        # Main loop
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate offspring
            Y = []
            for i in range(lam):
                z = rng.standard_normal(d)
                y = x_mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                Y.append(y)

            # Evaluate offspring
            F = []
            for y in Y:
                fy = safe_eval(y)
                if fy is None:
                    F.append(np.inf)
                else:
                    F.append(fy)

            if calls >= budget:
                break

            # Sort by fitness
            idx = np.argsort(F)
            Y_sorted = [Y[i] for i in idx]
            F_sorted = [F[i] for i in idx]

            # Update mean
            y_w = np.sum([w[i] * Y_sorted[i] for i in range(mu)], axis=0)
            x_mean_old = x_mean.copy()
            x_mean = y_w

            # Update sigma
            z_w = invsqrtC @ (y_w - x_mean_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * z_w
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls - budget + 1))) < (1.4 + 2 / (d + 1)) * chi_n
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (y_w - x_mean_old) / sigma

            # Update C
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank1 = np.outer(p_c, p_c)
            rankmu = np.sum([w[i] * np.outer(Y_sorted[i] - x_mean_old, Y_sorted[i] - x_mean_old) for i in range(mu)], axis=0)
            C = (1 + c_1 * delta_h + c_mu - c_1) * C + c_1 * rank1 + c_mu * rankmu

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update square root and inverse
            update_matrices()

            # Ensure x_mean is in bounds
            x_mean = np.clip(x_mean, -5, 5)

            # Evaluate new mean if we have budget
            if calls < budget:
                fm = safe_eval(x_mean)

        return best_x if best_x is not None else x_mean, best_f
# EVOLVE-BLOCK-END
