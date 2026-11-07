"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer combining (1+1)-ES for initial exploration and
    CMA-ES for exploitation, with a time guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        if budget < 20:
            rng = np.random.default_rng(0)
            best_x = rng.uniform(-5, 5, n)
            best_f = func(best_x)
            for _ in range(budget - 1):
                x = rng.uniform(-5, 5, n)
                f = func(x)
                if f < best_f:
                    best_f, best_x = f, x
            return best_x, best_f

        # Phase 1: (1+1)-ES for quick convergence
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, n)
        fx = func(x)
        sigma = 2.0
        evals = 1
        best_x = x.copy()
        best_f = fx

        # Run (1+1)-ES for 50% of the budget or until time limit
        phase1_budget = budget // 2
        for _ in range(phase1_budget - 1):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            evals += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
            else:
                sigma *= 0.87

        # Phase 2: CMA-ES for fine-tuning
        remaining_budget = budget - evals
        if remaining_budget > 20 and time.time() < deadline:
            # Initialize CMA-ES with current best
            m = best_x.copy()
            sigma = max(sigma, 0.1)
            C = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)

            # CMA-ES parameters
            lam = 4 + int(3 * np.log(n))
            if lam < 4:
                lam = 4
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mueff = 1.0 / np.sum(w**2)

            c_sigma = (mueff + 2) / (n + mueff + 5)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma

            while evals < budget and time.time() < deadline:
                # Get current C^(-1/2)
                D, B = np.linalg.eigh(C)
                D = np.maximum(D, 1e-10)
                C_inv_sqrt = (B / np.sqrt(D)) @ B.T

                # Sample offspring
                Y = np.zeros((lam, n))
                for i in range(lam):
                    if evals >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    Y[i] = m + sigma * (C_inv_sqrt @ z)
                    F = np.empty(lam)
                    F[i] = func(Y[i])
                    evals += 1
                    if F[i] < best_f:
                        best_f, best_x = F[i], Y[i].copy()

                if evals >= budget or time.time() > deadline:
                    break

                # Complete remaining offspring if budget allows
                for i in range(i + 1, lam):
                    if evals >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    Y[i] = m + sigma * (C_inv_sqrt @ z)
                    F[i] = func(Y[i])
                    evals += 1
                    if F[i] < best_f:
                        best_f, best_x = F[i], Y[i].copy()

                if evals >= budget:
                    break

                # Sort
                order = np.argsort(F)
                Y_best = Y[order[:mu]]
                F_best = F[order[:mu]]

                # Update mean
                m_old = m.copy()
                m = w @ Y_best.T

                # Update path statistics
                m_diff = m - m_old
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ m_diff)

                hc = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (evals // lam))) <
                      (1.4 + c_c / n) * np.sqrt((n + 1) / (n - 1)))
                pc = (1 - c_c) * pc + hc * np.sqrt(c_c * (2 - c_c) * mueff) * (C_inv_sqrt @ m_diff)

                # Update covariance matrix
                rank_one = pc @ pc.T + (1 - hc) * c_sigma * (2 - c_sigma) * C
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    y_i = C_inv_sqrt @ (Y_best[i] - m_old)
                    rank_mu += w[i] * np.outer(y_i, y_i)

                C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(2 * n) - 1))

        return best_x, best_f
# EVOLVE-BLOCK-END
