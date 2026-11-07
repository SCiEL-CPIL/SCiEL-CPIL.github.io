"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive restarts for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s limit

        # Wrap func to count calls and enforce budget
        call_count = [0]
        max_calls = budget

        def counted_func(x):
            if call_count[0] >= max_calls:
                return float('inf')
            call_count[0] += 1
            x = np.clip(x, -5, 5)
            val = func(x)
            return val

        # CMA-ES parameters
        # Standard CMA-ES with population size lambda
        lambda_pop = 4 + int(3 * np.log(dim))  # ~14 for dim=5
        # For 500 budget, we can do about 500/14 ≈ 35 generations
        # But we need to be careful with time

        # Initial parameters
        x_best = None
        f_best = float('inf')

        # We'll do multiple restarts with different starting points
        # Use a simple strategy: run CMA-ES for a portion of budget, restart if stuck

        # For robustness, let's use a well-tuned CMA-ES
        # Population size
        lam = max(4, 4 + int(3 * np.log(dim)))  # ~14 for dim=5
        # Step size
        sigma0 = 2.0

        # We'll use a simplified but effective CMA-ES
        # Key idea: maintain a distribution N(mu, sigma^2 * C)
        # For simplicity and speed, we'll use diagonal covariance initially,
        # then adapt based on success/failure

        # Actually, let me implement a proper but compact CMA-ES
        # with the standard update rules

        # CMA-ES implementation
        def cma_es(func_eval, x0, sigma, budget_remaining, dim, deadline):
            """Run CMA-ES with given initial conditions."""
            nonlocal call_count, f_best, x_best

            # CMA-ES parameters
            lam = max(4, 4 + int(3 * np.log(dim)))
            mu = lam // 2  # number of recombination weights
            # Weights: linearly decreasing
            w = np.log(np.arange(1, mu + 1)) - np.log(mu)
            w /= w.sum()
            mueff = 1.0 / np.sum(w**2)

            # Update rates
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))

            # Initial values
            x_mean = x0.copy()
            sigma = sigma
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

            best_x = x_mean.copy()
            best_f = func_eval(x_mean)
            call_count[0] += 1  # Count the initial evaluation
            if best_f < f_best:
                f_best = best_f
                x_best = best_x.copy()

            gen = 0
            while call_count[0] < budget and time.time() < deadline:
                # Generate population
                ys = []
                for i in range(lam):
                    # Sample from N(0, C)
                    z = np.random.randn(dim)
                    # Transform: C^{1/2} z
                    y = x_mean + sigma * (C_inv_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    ys.append(y)

                # Evaluate
                fs = []
                for y in ys:
                    fy = func_eval(y)
                    fs.append(fy)
                    if fy < f_best:
                        f_best = fy
                        best_x = y.copy()
                        x_best = best_x.copy()

                # Sort
                idx = np.argsort(fs)
                ys = [ys[i] for i in idx]
                fs = [fs[i] for i in idx]

                # Update mean
                x_mean_new = sum(w[i] * ys[i] for i in range(mu))

                # Update sigma (CSA)
                # Compute normalized step
                step = (x_mean_new - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ step
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 1 / dim) * chi_n:
                    sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update C (CSA for C)
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * step
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = (ys[i] - x_mean) / sigma
                    rank_mu += w[i] * np.outer(y_i, y_i)

                C = (1 + c_1 + c_mu - c_c) * C + c_c * (np.outer(p_c, p_c) - C) + c_mu * rank_mu

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                # Regularize if needed
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    if np.min(eigvals) < 1e-10:
                        eigvals = np.maximum(eigvals, 1e-10)
                        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                except:
                    C = np.eye(dim)

                # Update C_inv_sqrt
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(np.maximum(eigvals, 1e-10))) @ eigvecs.T
                except:
                    C_inv_sqrt = np.eye(dim)

                x_mean = x_mean_new
                gen += 1

            return best_x, best_f

        # Strategy: multiple restarts
        # Use different initial points and allocate budget across restarts
        # For 500 calls, we can do several restarts

        # Number of restarts based on budget
        n_restarts = min(5, budget // 50)  # At least 1, at most 5
        calls_per_restart = budget // n_restarts

        # Initial points: try center and a few random points
        rng = np.random.default_rng(42)

        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            if call_count[0] >= budget:
                break

            if restart == 0:
                x0 = np.zeros(dim)
                sigma0 = 2.0
            else:
                # Random restart around previous best or random
                if x_best is not None:
                    x0 = x_best + 1.0 * rng.standard_normal(dim)
                    x0 = np.clip(x0, -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, dim)
                sigma0 = 1.0

            # Run CMA-ES for this restart
            cma_es(counted_func, x0, sigma0, budget, dim, deadline)

        return x_best, f_best
# EVOLVE-BLOCK-END
