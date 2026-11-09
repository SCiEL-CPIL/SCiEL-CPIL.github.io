"""Rewritten optimizer: multi-start CMA-ES with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


class Optimizer:
    """Multi-start CMA-ES for Rastrigin-rotated f15 on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270  # safety margin before 300s limit

        best_x = np.full(dim, np.nan)
        best_f = np.inf

        # CMA-ES implementation
        def cmaes(func, x0, sigma0, budget_cma, dim, deadline):
            n = dim
            m = max(4, int(round((4 + int(np.log(n))) * n)))  # population size
            # CMA-ES parameters
            cmac = 1.0 / (n + 4)
            cmam = 1.0 / (n + 4)
            c1 = (2 / ((n + 1.3) ** 2 + m)) * (1 - cmac)
            cmu = min(0.75 - c1, 2 * (2 - (n + 2) / (n + 4)) / ((n + 2) + m))
            cs = (m + 1.7) / (n + m + 5)
            ds = 1 + 2 * max(0, np.sqrt((m - 1) / (n + 1)) - 1) + cs
            cc = (4 + n / m) / (4 + n + m / n)
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            # Initialize
            x = np.array(x0, dtype=float)
            sigma = sigma0
            pc = np.zeros(n)
            ps = np.zeros(n)
            B = np.eye(n)
            D = np.ones(n)
            C = np.eye(n)

            # Initial function eval
            fx = func(np.clip(x, -5, 5))
            best_local_x = x.copy()
            best_local_f = fx

            count = 1

            for gen in range(budget_cma):
                if time.time() > deadline:
                    break

                # Sample population
                w = np.zeros(m)
                w[:m // 2] = np.log(m / 2) - np.log(np.arange(1, m // 2 + 1))
                w /= w.sum()
                mu = m // 2

                # Generate offspring
                ys = []
                for i in range(m):
                    if time.time() > deadline:
                        break
                    z = np.random.standard_normal(n)
                    # Transform using C
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    ys.append((y, z))
                    count += 1

                # Evaluate
                fvals = []
                for y, z in ys:
                    fy = func(y)
                    fvals.append(fy)
                    if fy < best_local_f:
                        best_local_f = fy
                        best_local_x = y.copy()

                # Sort by fitness
                order = np.argsort(fvals)
                ys_sorted = [ys[i] for i in order]
                fvals_sorted = [fvals[i] for i in order]

                # Update mean
                x_old = x.copy()
                x_new = sum(w[i] * ys_sorted[i][0] for i in range(mu))

                # Update step size control
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (B @ (D ** -1)) @ ((x_new - x_old) / sigma)
                pc = (1 - cmac) * pc + cmac * ((1 - cmam) * pc + np.sqrt(cmac * (2 - cmac) * mu_eff) * (x_new - x_old) / sigma)

                # Update covariance
                delta_sig = np.linalg.norm(ps) / chi_n - 1
                hsig = 1.0 if (delta_sig ** 2) / (1 - (1 - cs) ** (2 * (gen + 1))) < (1 + 2 / (n + 1)) * (1 - (1 - cs) ** (2 * (gen + 1))) else 0.0
                rank1 = np.outer(pc, pc)
                rankmu = sum(w[i] * np.outer(ys_sorted[i][1], ys_sorted[i][1]) for i in range(mu))
                C = (1 + c1 * hsig + cmu * hsig - c1 - cmu) * C + c1 * rank1 + cmu * rankmu

                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))

                # Update B and D
                C_ = 0.5 * (C + C.T)  # symmetrize
                D, B = np.linalg.eigh(C_)
                D = np.sqrt(np.maximum(D, 1e-10))
                B = B[:, ::-1]
                D = D[::-1]

                x = x_new
                fx = fvals_sorted[0]

                if count >= budget_cma:
                    break

            return best_local_x, best_local_f, count

        # Multi-start strategy
        np.random.seed(42)

        # First start: center
        x0 = np.zeros(dim)
        sigma0 = 2.0
        budget_per_start = 120

        # Determine number of starts based on budget
        n_starts = max(1, budget // budget_per_start)
        actual_budget_per = budget // n_starts

        total_calls = 0
        for s in range(n_starts):
            if time.time() > deadline:
                break
            if total_calls >= budget:
                break

            if s == 0:
                x0 = np.zeros(dim)
            else:
                # Random restarts biased toward center
                x0 = np.random.uniform(-3, 3, dim)

            sigma0 = 2.0 if s == 0 else 1.5

            remaining = budget - total_calls
            budget_this = min(actual_budget_per, remaining)
            if budget_this < 10:
                break

            res_x, res_f, calls_used = cmaes(func, x0, sigma0, budget_this, dim, deadline)
            total_calls += calls_used

            if res_f < best_f:
                best_f = res_f
                best_x = res_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
