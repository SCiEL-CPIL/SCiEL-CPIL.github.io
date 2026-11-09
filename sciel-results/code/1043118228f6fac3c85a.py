"""Optimizer for BBOB-style Rosenbrock (f8) on [-5,5]^5."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for Rosenbrock optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280  # 280s safety margin before 330s kill

        # CMA-ES parameters
        lambda_ = 9  # population size
        mu = lambda_ // 2

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.dot(w, w)

        # Strategy parameters
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_amps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_mu
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial point: center of box
        x = np.zeros(dim)
        sigma = 2.0  # initial step size

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)

        # Evaluate initial point
        best_x = x.copy()
        best_f = func(x)
        calls = 1

        for gen in range(budget - 1):
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            Y = []
            for i in range(lambda_):
                z = np.random.standard_normal(dim)
                Y.append(z)
            Y = np.array(Y)

            # Sort by fitness (evaluate all)
            X = []
            F = []
            for i in range(lambda_):
                xi = x + sigma * Y[i]
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                X.append(xi)
                F.append(fi)
                calls += 1
                if time.time() > deadline:
                    break

            if calls >= budget:
                break

            # If we ran out of budget mid-generation, break
            if len(X) < lambda_:
                # Update best
                for i in range(len(X)):
                    if F[i] < best_f:
                        best_f = F[i]
                        best_x = X[i].copy()
                break

            # Sort
            idx = np.argsort(F)
            X_sorted = np.array(X)[idx]
            F_sorted = np.array(F)[idx]

            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()

            # Selection
            Y_w = np.array(Y)[idx]
            X_w = np.array(X)[idx]

            # Weighted mean
            m_new = np.sum(w[:mu] * X_w[:mu], axis=0)

            # Update CMA
            # Difference vectors
            diff = Y_w[:mu] - np.mean(Y_w, axis=0)  # approximation
            # Actually use the standard CMA-ES update

            # Move distribution
            p_sigma = (1 - c_c) * p_sigma + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (m_new - x)
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_c) ** (2 * (gen + 1)))) / chi_n < (1.4 + 2 / (dim + 1))

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - x) / sigma

            # Rank-one update
            y_mean = (m_new - x) / sigma
            C = (1 + c_1 * (h_sigma * d_amps - 1) - c_mu) * C + c_1 * np.outer(y_mean, y_mean)

            # Rank-mu update
            for i in range(mu):
                yi = (X_w[i] - x) / sigma
                C -= c_mu * w[i] * np.outer(yi, yi)

            # Symmetrize
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(0.5 / d_amps * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update mean
            x = m_new
            x = np.clip(x, -5, 5)

        return best_x, best_f
