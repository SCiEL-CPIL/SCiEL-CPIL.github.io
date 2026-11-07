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
        timeout = 280.0  # Safety margin for 300s limit

        d = self.dim
        budget = self.budget

        # Best tracking
        best_x = np.zeros(d)
        best_f = np.inf

        # IPOP-CMA-ES with restarts
        # Start with larger population, reduce on stagnation
        lambda_ = 10
        min_lambda = 4
        restart_count = 0
        max_restarts = 50
        stagnation_threshold = 0.001  # Relative improvement threshold
        last_improvement_eval = 0
        last_best_f = np.inf

        # For the first restart, use a reasonable initial sigma
        initial_sigma = 2.0

        while restart_count < max_restarts and budget > 0 and time.time() - start_time < timeout:
            # Set up CMA-ES for this restart
            m = np.zeros(d)
            sigma = initial_sigma
            if restart_count > 0:
                # Perturb the initial mean around the best found
                m = best_x + np.random.randn(d) * sigma * 0.5

            # Population size for this restart
            lambda_ = max(min_lambda, int(4 + 3 * np.log(d) + restart_count * 0.5))
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

            chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (32 * d * d))

            # Evaluate initial mean
            f_m = func(m)
            budget -= 1
            if f_m < best_f:
                best_f = f_m
                best_x = m.copy()

            # Run CMA-ES until budget exhausted or stagnation
            gen = 0
            while budget >= lambda_ and time.time() - start_time < timeout:
                gen += 1

                # Generate population
                X = []
                for i in range(lambda_):
                    z = np.random.randn(d)
                    y = m + sigma * B @ (D * z)
                    y = np.clip(y, -5, 5)
                    X.append(y)

                # Evaluate
                F = np.array([func(x) for x in X])
                budget -= lambda_

                # Update best
                idx_best = np.argmin(F)
                if F[idx_best] < best_f:
                    best_f = F[idx_best]
                    best_x = X[idx_best].copy()
                    last_improvement_eval = -budget  # Track when improvement happened
                    last_best_f = best_f

                # Sort
                idx = np.argsort(F)
                x_sorted = [X[i] for i in idx]

                # Weighted mean
                m_new = np.zeros(d)
                for i in range(mu):
                    m_new += weights[i] * x_sorted[i]

                # Step size control
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ ((m_new - m) / sigma))

                # Heuristic to adapt sigma based on pc length
                hsig = np.linalg.norm(pc) / chi_d
                if hsig < (1.4 + 2 / (d + 1)):
                    sigma *= (1 + c_sigma * 0.3)
                    sigma *= np.exp(c_sigma * (hsig - 1))
                else:
                    sigma /= (1 + c_sigma)

                # Update C
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(pc, pc) + (1 - (1 - c_c) ** 2) * C)

                for i in range(mu):
                    diff = x_sorted[i] - m_new
                    C += c_mu * weights[i] * np.outer(diff, diff)

                # Ensure C is positive definite and symmetric
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C = (eigvecs * eigvals) @ eigvecs.T
                    B = eigvecs
                    D = np.sqrt(eigvals)
                except np.linalg.LinAlgError:
                    B = np.eye(d)
                    D = np.ones(d)
                    C = np.eye(d)

                m = m_new

                # Check for stagnation - if no improvement for many generations, restart
                if gen > 10 and (last_best_f - best_f) / (abs(last_best_f) + 1e-8) < stagnation_threshold:
                    if gen - last_improvement_eval > 20:
                        break

            restart_count += 1
            # Reduce sigma for next restart to explore finer scales
            initial_sigma *= 0.7

            # If we have very little budget left, do (1+1)-ES steps
            if budget > 0 and budget < lambda_:
                for _ in range(budget):
                    if time.time() - start_time > timeout:
                        break
                    y = best_x + np.random.randn(d) * 0.5
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    budget -= 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
