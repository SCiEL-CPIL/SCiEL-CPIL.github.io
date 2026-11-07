"""Rewritten optimizer: CMA-ES with adaptive restarts for multimodal BBOB functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and final local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_exceeded():
            return time.time() > deadline

        def cma_es(x0, sigma0, max_evals, rng, func):
            """Run CMA-ES from x0 with sigma0 for max_evals evaluations."""
            x = x0.copy()
            fx = func(x)
            sigma = sigma0
            best_x = x.copy()
            best_f = fx
            calls = 1

            # CMA-ES parameters
            mu = max(1, (d + 1) // 2)  # number of parents
            lambda_ = 2 * mu  # population size
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
            c_1 = (2 - c_c + mu_eff / (d + 2)) / ((d + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, (mu_eff - 2 + 2 / (d + mu_eff)) / ((d + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

            p_sigma = np.zeros(d)
            p_c = np.zeros(d)
            C = np.eye(d)
            eigenvalues = np.ones(d)
            eigenvectors = np.eye(d)
            eigenvalue_counter = 0
            damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma

            def update_eigen():
                nonlocal eigenvalues, eigenvectors
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.clip(eigenvalues, 1e-20, None)

            update_eigen()

            for _ in range(max_evals - 1):
                if time_exceeded():
                    break
                if calls + lambda_ > max_evals:
                    break

                # Sample population
                ys = np.zeros((lambda_, d))
                xs = np.zeros((lambda_, d))
                fs = np.zeros(lambda_)
                for i in range(lambda_):
                    ys[i] = rng.standard_normal(d)
                    xs[i] = np.clip(x + sigma * ys[i], -5, 5)
                    fs[i] = func(xs[i])
                    calls += 1

                # Sort by fitness
                order = np.argsort(fs)
                ys = ys[order]
                xs = xs[order]
                fs = fs[order]

                # Update best
                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = xs[0].copy()

                # Compute weighted mean
                m = np.sum(weights[:, None] * ys[:mu], axis=0)

                # Update step size control
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                    np.sqrt(np.diag(C)) / sigma * m
                )

                hsig = (
                    np.sqrt((1 - (1 - c_sigma) ** (2 * (calls - lambda_))) / (1 - (1 - c_sigma) ** 2))
                    * chi_n
                    / (1 - (1 - c_sigma) ** (calls - lambda_))
                )
                hsig = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - lambda_))) < (1.4 + 2 / (d + 1)) * chi_n else 0.0

                # Update evolution path and covariance matrix
                p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * m
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + \
                    c_1 * np.outer(p_c, p_c) + \
                    c_mu / d * np.sum(weights[:, None, None] * np.outer(ys[:mu], ys[:mu]), axis=0)

                eigenvalue_counter += 1
                if eigenvalue_counter >= 1 / ((d + 1) * (c_1 + c_mu)):
                    update_eigen()
                    eigenvalue_counter = 0

                sigma *= np.exp(c_sigma / damps * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-15, 10.0)

                # Update mean
                x = np.clip(x + sigma * m, -5, 5)
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            return best_x, best_f, calls

        # Phase 1: Initial random sampling
        n_random = min(15, budget // 20)
        candidates = []
        for i in range(n_random):
            if time_exceeded():
                break
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            candidates.append((x, fx))

        candidates.sort(key=lambda t: t[1])
        best_x, best_f = candidates[0]
        calls_used = len(candidates)

        # Phase 2: CMA-ES with restarts
        remaining = budget - calls_used
        n_restarts = min(4, max(1, remaining // 60))
        budget_per_restart = remaining // n_restarts

        for restart in range(n_restarts):
            if time_exceeded():
                break
            if calls_used >= budget:
                break

            if restart == 0:
                x0 = best_x.copy()
                sigma0 = 1.0
            elif restart == 1 and len(candidates) > 1:
                x0 = candidates[1][0].copy()
                sigma0 = 0.8
            else:
                x0 = best_x + rng.standard_normal(d) * 0.3
                x0 = np.clip(x0, -5, 5)
                sigma0 = 0.5

            new_x, new_f, calls_made = cma_es(x0, sigma0, min(budget_per_restart, budget - calls_used), rng, func)
            calls_used += calls_made

            if new_f < best_f:
                best_f = new_f
                best_x = new_x.copy()

        # Phase 3: Local refinement
        remaining = budget - calls_used
        if remaining > 5 and not time_exceeded():
            x = best_x.copy()
            fx = best_f
            sigma = 0.2
            for _ in range(min(remaining, 30)):
                if time_exceeded():
                    break
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                sigma = np.clip(sigma, 1e-6, 1.0)
                if sigma < 1e-5:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
