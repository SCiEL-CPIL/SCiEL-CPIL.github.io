"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for Büche-Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 280.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))  # population size
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

        def cma_es(x0, sigma0, max_iter, deadline):
            x = x0.copy()
            sigma = sigma0
            ps = np.zeros(n)
            hs = np.zeros(n)
            B = np.eye(n)
            D = 1.0
            pc = np.zeros(n)
            best_x = x.copy()
            best_f = func(x)
            calls = 1

            for it in range(max_iter):
                if time.time() > deadline:
                    break
                # Generate population
                offspring = []
                for i in range(lambda_):
                    z = np.random.standard_normal(n)
                    y = x + sigma * (B @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls += 1
                    offspring.append((fy, y))
                if calls >= budget:
                    break
                # Sort
                offspring.sort(key=lambda t: t[0])
                # Update best
                if offspring[0][0] < best_f:
                    best_f = offspring[0][0]
                    best_x = offspring[0][1].copy()
                # Weighted mean
                xnew = np.sum([weights[i] * offspring[i][1] for i in range(mu)], axis=0)
                # Step size control
                ydiff = (xnew - x) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / D * ydiff
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (it + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1))
                hs = hsig
                pc = (1 - cc) * pc + hs * cc / D * ydiff
                # Covariance matrix adaptation
                rank1 = np.outer(pc, pc)
                delta_h = (1 - hs) * cc * (1 - cc)
                rankmu = np.zeros((n, n))
                for i in range(mu):
                    zi = (offspring[i][1] - x) / sigma
                    rankmu += np.outer(weights[i] * zi, weights[i] * zi)
                C = (1 + c1 * delta_h - c1) * C + c1 * rank1 + cmu * rankmu
                # Eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                D = np.sqrt(eigvals)
                B = eigvecs
                # Update
                x = xnew
                sigma *= np.exp(cs / (n + 4) * (np.linalg.norm(ydiff) / np.sqrt(2 * n) - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)
            return best_x, best_f, calls

        # Restart strategy
        best_x = None
        best_f = np.inf
        total_calls = 0
        remaining = budget

        # Initial population-based exploration
        n_restarts = 3
        calls_per_restart = remaining // n_restarts

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if total_calls >= budget:
                break

            # Different starting points for each restart
            if r == 0:
                x0 = np.zeros(n)
                sigma0 = 2.0
            elif r == 1:
                x0 = np.random.uniform(-3, 3, n)
                sigma0 = 1.5
            else:
                x0 = np.random.uniform(-4, 4, n)
                sigma0 = 3.0

            max_iter = min(calls_per_restart // lambda_, 50)
            x_best, f_best, calls_used = cma_es(x0, sigma0, max_iter, deadline)
            total_calls += calls_used

            if f_best < best_f:
                best_f = f_best
                best_x = x_best

            # Local refinement around best point
            if best_x is not None and time.time() < deadline and total_calls < budget:
                # Simple local search
                local_sigma = 0.1
                for _ in range(min(20, budget - total_calls)):
                    if time.time() > deadline:
                        break
                    y = np.clip(best_x + local_sigma * np.random.standard_normal(n), -5, 5)
                    fy = func(y)
                    total_calls += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y
                    local_sigma *= 0.95

        # Final polish
        if best_x is not None and time.time() < deadline and total_calls < budget:
            for _ in range(min(30, budget - total_calls)):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + 0.01 * np.random.standard_normal(n), -5, 5)
                fy = func(y)
                total_calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y

        return best_x, best_f
# EVOLVE-BLOCK-END
