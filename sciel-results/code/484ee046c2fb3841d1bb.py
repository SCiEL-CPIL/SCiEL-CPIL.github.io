"""Rewritten program using CMA-ES for robust multimodal optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 285.0  # Safety margin for 300s limit

        d = self.dim
        budget = self.budget

        # CMA-ES parameters
        m = np.zeros(d)
        sigma = 1.5  # Slightly smaller initial step to focus on the box

        # Population size
        lambda_ = 4 + int(3 * np.log(d))
        if lambda_ < 4:
            lambda_ = 4

        # CMA-ES adaptation rates
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((d + 2) ** 2 + mu_eff))

        # State variables
        pc = np.zeros(d)
        B = np.eye(d)
        D = np.ones(d)
        C = np.eye(d)

        # Initial evaluation
        best_x = m.copy()
        best_f = func(m)
        evals_used = 1
        remaining = budget - 1

        # Chi-d approximation
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (32 * d * d))

        # Generation counter for time checks
        gen = 0

        while remaining > 0 and time.time() - start_time < timeout:
            gen += 1

            # If we can't fit a full population, do (1+1)-ES steps
            if remaining < lambda_:
                for _ in range(remaining):
                    if time.time() - start_time > timeout:
                        break
                    y = m + sigma * np.random.randn(d)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    evals_used += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    # Simple step size adaptation
                    if fy < best_f:
                        sigma *= 1.05
                    else:
                        sigma *= 0.95
                break

            # Generate population
            X = []
            for i in range(lambda_):
                z = np.random.randn(d)
                y = m + sigma * B @ (D * z)
                y = np.clip(y, -5, 5)
                X.append(y)

            # Evaluate
            F = np.array([func(x) for x in X])
            evals_used += lambda_
            remaining -= lambda_

            # Update best
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = X[idx_best].copy()

            # Sort
            idx = np.argsort(F)
            x_sorted = [X[i] for i in idx]
            f_sorted = F[idx]

            # Weighted mean
            m_new = np.zeros(d)
            for i in range(mu):
                m_new += weights[i] * x_sorted[i]

            # Step size control (CSA)
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ ((m_new - m) / sigma))

            # Heuristic to adapt sigma based on pc length
            if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (2 * (evals_used / lambda_))) < (1.4 + 2 / (d + 1)) * chi_d:
                sigma /= (1 + c_sigma)
            else:
                sigma *= (1 + c_sigma * 0.3)
                sigma *= np.exp(c_sigma * (np.linalg.norm(pc) / chi_d - 1))

            # Update C
            delta_h = 1.0 if np.linalg.norm(pc) / chi_d < (1.4 + 2 / (d + 1)) else 0.0

            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + (1 - (1 - c_c) ** 2) * C)

            for i in range(mu):
                diff = x_sorted[i] - m_new
                C += c_mu * weights[i] * np.outer(diff, diff)

            # Ensure C is positive definite and symmetric
            C = (C + C.T) / 2
            # Eigendecomposition for stability
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues
                eigvals = np.clip(eigvals, 1e-12, None)
                C = (eigvecs * eigvals) @ eigvecs.T
                B = eigvecs
                D = np.sqrt(eigvals)
            except np.linalg.LinAlgError:
                B = np.eye(d)
                D = np.ones(d)
                C = np.eye(d)

            m = m_new

        return best_x, best_f
# EVOLVE-BLOCK-END
