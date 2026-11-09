"""CMA-ES optimizer for BBOB f12 (Bent Cigar) with adaptive restarts."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with adaptive restarts for ill-conditioned functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin
        max_calls = budget

        calls = [0]
        best_x = None
        best_fx = 1e18

        def f(x):
            nonlocal best_x, best_fx
            calls[0] += 1
            val = float(func(x))
            if not np.isfinite(val):
                val = 1e18
            if val < best_fx:
                best_fx = val
                best_x = x.copy()
            return val

        def remaining():
            return deadline - time.time()

        def cma_es(x0, fx0, max_iter):
            """Run CMA-ES from x0 with initial fx0."""
            n = dim
            # Parameters
            lambda_ = min(4 + int(3 * np.log(n)), 20)  # population size
            mu = lambda_ // 2
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            w = np.concatenate([w, np.zeros(lambda_ - mu)])
            mu_eff = 1.0 / np.sum(w[:mu] ** 2)

            # Strategy parameters
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # State
            x = x0.copy()
            sigma = 1.0
            C = np.eye(n)
            C_sqrt = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            h_sigma = 0.0

            it = 0
            no_improve = 0
            best_local_x = x0.copy()
            best_local_fx = fx0

            while it < max_iter and calls[0] < max_calls and remaining() > 5.0:
                it += 1
                # Generate population
                Y = []
                fvals = []
                for k in range(lambda_):
                    # Sample from N(0, C)
                    z = np.random.randn(n)
                    y = x + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    Y.append(y)
                    fvals.append(f(y))

                # Sort by fitness
                order = np.argsort(fvals)
                Y_sorted = [Y[i] for i in order]
                fvals_sorted = [fvals[i] for i in order]

                # Update best
                if fvals_sorted[0] < best_local_fx:
                    best_local_fx = fvals_sorted[0]
                    best_local_x = Y_sorted[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Update mean
                new_x = np.zeros(n)
                for i in range(mu):
                    new_x += w[i] * Y_sorted[i]
                x = new_x

                # Update sigma
                y_sigma = (C_sqrt @ (w @ np.array(Y_sorted[:mu]))) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1)))) / (1 - c_sigma) / chi_n
                h_sigma = min(h_sigma, 1.0)

                # Update covariance
                y_c = (C_sqrt @ (w @ np.array(Y_sorted[:mu]))) / sigma
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    rank_mu += w[i] * np.outer(Y_sorted[i] - x, Y_sorted[i] - x) / (sigma ** 2)

                C = ((1 + c_c * (2 - c_c) - c_sigma) * C + c_sigma * (rank_one + (1 - h_sigma * c_sigma * d_sigma) * rank_mu - C))
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(y_sigma) / chi_n - 1))

                # Ensure positive definiteness
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C = (eigvecs @ np.diag(eigvals)) @ eigvecs.T
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                except:
                    C = np.eye(n)
                    C_sqrt = np.eye(n)

                # Convergence / restart check
                if sigma < 1e-12 or no_improve > 50 or np.max(np.abs(x)) > 100:
                    break

            return best_local_x, best_local_fx

        # Multi-start CMA-ES with different initializations
        rng = np.random.default_rng(42)
        starts = [
            np.zeros(dim),
            rng.uniform(-3, 3, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-2, 2, dim),
        ]

        for s in starts:
            if calls[0] >= max_calls or remaining() < 10.0:
                break
            s = np.clip(s, -5, 5)
            fx = f(s)
            x, fx = cma_es(s, fx, max_iter=300)

        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = f(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
