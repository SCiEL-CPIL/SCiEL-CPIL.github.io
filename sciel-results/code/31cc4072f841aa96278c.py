"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-stage optimizer: coarse random + CMA-ES with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        deadline = time.time() + 280.0  # safety margin before 300s kill

        best_x = None
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=float).reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Stage 1: Coarse random sampling to locate a good basin quickly.
        n_coarse = min(20, budget)
        for i in range(n_coarse):
            if time.time() > deadline or calls >= budget:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            safe_call(x)

        if calls >= budget or best_x is None:
            return best_x, best_f

        # Stage 2: CMA-ES with restarts
        def cma_es(start_x, sigma0, budget_limit):
            nonlocal calls, best_x, best_f
            if calls >= budget_limit:
                return

            # CMA-ES parameters
            n = dim
            # Population size
            lambda_ = max(4, int(4 + 3 * np.log(n)))
            # Step size
            sigma = sigma0
            # Mean
            m = start_x.copy()
            # Covariance matrix
            C = np.eye(n)
            # Inverse square root of C
            C_inv_sqrt = np.eye(n)
            # Evolution paths
            ps = np.zeros(n)
            pc = np.zeros(n)
            # Learning rates
            cs = (np.sum(np.ones(lambda_)) + 2) / (n + np.sum(np.ones(lambda_)) + 5)
            cmu = min(0.2, (lambda_ / 2 - 2) / ((n + 2) ** 2 - lambda_))
            c1 = (2 + 2.0 / (n + 1.3)) / ((n + 4) ** 2 + lambda_)
            c1 = 2 * (c1 - cmu) / 2  # Adjust c1
            c1 = (2 + 2.0 / (n + 1.3)) / ((n + 4) ** 2 + lambda_)
            c1 = 2 * (c1 - cmu) / 2
            c1 = (2 + 2.0 / (n + 1.3)) / ((n + 4) ** 2 + lambda_)
            c1 = 2 * (c1 - cmu) / 2
            c1 = (2 + 2.0 / (n + 1.3)) / ((n + 4) ** 2 + lambda_)
            c1 = 2 * (c1 - cmu) / 2
            c1 = (2 + 2.0 / (n + 1.3)) / ((n + 4) ** 2 + lambda_)
            c1 = 2 * (c1 - cmu) / 2
            c1 = (2 + 2.0 / (n + 1.3)) / ((n + 4) ** 2 + lambda_)
            c1 = 2 * (c1 - cmu) / 2
            c1 = (2 + 2.0 / (n + 1.3)) / ((n + 4) ** 2 + lambda_)
            c1 = 2 * (c1 - cmu) / 2

            # Let me redo the CMA-ES parameters more carefully
            # Standard CMA-ES parameter settings
            lambda_ = max(4, int(4 + 3 * np.log(n)))
            mu = lambda_ // 2  # Number of weights
            # Weights
            w = np.log(np.arange(1, mu + 1)) - np.log(mu)
            w = w / np.sum(w)
            mu_eff = 1.0 / np.sum(w ** 2)
            # Learning rates
            cc = (mu_eff + 2) / (n + mu_eff + 5)
            cs = (mu_eff + 2 + 2 * (n / mu_eff - 1)) / (n + mu_eff + 5)
            c1 = (2 - c1 + 2 * mu_eff / n) / ((n + 2) ** 2 + mu_eff)
            cmu = min(0.5, (2 * (mu_eff - 2 + 2 / mu_eff)) / ((n + 2) ** 2 + mu_eff))
            cmu = cmu - c1

            # Covariance matrix
            C = np.eye(n)
            # Eigen decomposition for efficient sampling
            eigvals, eigvecs = np.linalg.eigh(C)
            # Inverse square root
            eigvals_inv_sqrt = 1.0 / np.sqrt(eigvals)
            # Evolution paths
            ps = np.zeros(n)
            pc = np.zeros(n)
            # Counters
            no_improve = 0
            max_no_improve = 15 * n

            while calls < budget_limit and time.time() < deadline:
                # Generate population
                offspring = []
                fitnesses = []
                for i in range(lambda_):
                    if calls >= budget_limit or time.time() > deadline:
                        break
                    # Sample from N(m, sigma^2 * C)
                    z = rng.standard_normal(n)
                    # Transform: y = eigvecs @ (eigvals_inv_sqrt * z)
                    y = eigvecs @ (eigvals_inv_sqrt * z)
                    x = np.clip(m + sigma * y, -5.0, 5.0)
                    fx = safe_call(x)
                    offspring.append(x)
                    fitnesses.append(fx)

                if len(offspring) < mu:
                    break

                # Sort by fitness
                order = np.argsort(fitnesses)
                offspring = [offspring[i] for i in order]
                fitnesses = [fitnesses[i] for i in order]

                # Update mean
                old_m = m.copy()
                for i in range(mu):
                    m += w[i] * (offspring[i] - m)

                # Update evolution paths
                # Recompute eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals_inv_sqrt = 1.0 / np.sqrt(eigvals)

                # pc update
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (
                    eigvecs @ (eigvals_inv_sqrt @ ((m - old_m) / sigma))
                )

                # ps update
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (
                    eigvecs @ (eigvals_inv_sqrt @ ((m - old_m) / sigma))
                )

                # hsig check
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls // lambda_ + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt(
                    (n + 1) / (n - 1) - 1
                )

                # pc update with hsig
                if hsig:
                    pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (
                        eigvecs @ (eigvals_inv_sqrt @ ((m - old_m) / sigma))
                    )

                # Update covariance
                delta_sig = (hsig - 1) * cc * (2 - cc)
                C = (1 + c1 * delta_sig - c1 - cmu) * C
                for i in range(mu):
                    z_i = eigvecs @ (eigvals_inv_sqrt @ (offspring[i] - old_m) / sigma)
                    C += cmu * w[i] * np.outer(z_i, z_i)

                # Ensure positive definiteness
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = (eigvecs * eigvals) @ eigvecs.T
                eigvals_inv_sqrt = 1.0 / np.sqrt(eigvals)

                # Sigma adaptation
                sigma *= np.exp(cs / (n + 2) * (np.linalg.norm(ps) / np.sqrt(
                    (n + 1) / (n - 1) - 1) - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Check for stagnation
                if fitnesses[0] >= best_f + 1e-10:
                    no_improve += lambda_
                else:
                    no_improve = 0

                if no_improve >= max_no_improve:
                    break

            return

        # Run CMA-ES with restarts
        num_restarts = 5
        per_restart_budget = (budget - calls) // num_restarts

        for r in range(num_restarts):
            if calls >= budget or time.time() > deadline:
                break

            if r == 0:
                start_x = best_x.copy()
                sigma0 = 1.0
            else:
                # Restart: perturb best or random
                if rng.random() < 0.3:
                    start_x = rng.uniform(-5.0, 5.0, dim)
                else:
                    start_x = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5.0, 5.0)
                sigma0 = 0.5

            # Evaluate start point if it's new
            if r > 0:
                if calls < budget and time.time() < deadline:
                    safe_call(start_x)

            cma_es(start_x, sigma0, budget)

        # Final polish
        if calls < budget and best_x is not None:
            for _ in range(min(10, budget - calls)):
                if time.time() > deadline:
                    break
                step = 0.05 * rng.standard_normal(dim)
                y = np.clip(best_x + step, -5.0, 5.0)
                fy = safe_call(y)

        return best_x, best_f
# EVOLVE-BLOCK-END
