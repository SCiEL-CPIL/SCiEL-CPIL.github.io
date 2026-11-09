"""CMA-ES style optimizer for BBOB f5 (Linear slope)."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with population-based search and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s kill

        # Population size: lambda = 4 + 3*ln(dim) for dim=5 -> ~9
        lam = max(4, int(np.ceil(4 + 3 * np.log(dim))))

        # Initial mean: center of the box
        m = np.zeros(dim)
        f_m = func(m)
        best_x = m.copy()
        best_f = f_m
        calls = 1

        # Initial sigma
        sigma = 1.5

        # CMA-ES parameters
        mu = lam // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            population = []
            f_values = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = m + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                population.append(y)
                f_values.append(fy)
                if calls >= budget:
                    break

            if calls >= budget:
                break

            # Sort by fitness
            order = np.argsort(f_values)
            population = [population[i] for i in order]
            f_values = [f_values[i] for i in order]

            # Compute weighted mean of parents
            m_new = sum(weights[i] * population[i] for i in range(mu))

            # Update evolution paths
            # ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mu_eff) * (B/D) @ ((m_new - m)/sigma)
            diff = (m_new - m) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ diff / D)

            # Update pc
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * max(1, calls - dim)))) < (1.4 + 2 / (dim + 1)) * chi
            pc = (1 - c_c) * pc + h_sigma * c_c * (m_new - m) / sigma

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            outer = np.outer(pc, pc)
            C = (1 + c_1 * delta_h - c_1) * C if hasattr(self, '_C') else None
            # Initialize C properly
            if not hasattr(self, '_C_initialized'):
                C = np.eye(dim)
                self._C_initialized = True
            else:
                C = getattr(self, '_C', np.eye(dim))
            
            # Actually let's just do a simpler approach: use diagonal CMA-ES
            # For dim=5, full CMA-ES is fine but let's keep it simple and robust
            # Use diagonal covariance
            C = getattr(self, '_C', np.eye(dim))
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (outer + (1 - h_sigma) * c_c * (2 - c_c) * np.eye(dim) - outer)
            C += c_mu * sum(weights[i] * np.outer(population[i] - m, population[i] - m) for i in range(mu))
            self._C = C

            # Update B and D via eigenvalue decomposition
            try:
                D2, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D2, 1e-20))
            except np.linalg.LinAlgError:
                D = np.ones(dim)
                B = np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi - 1))
            sigma = max(1e-10, min(sigma, 50))

            m = m_new
            f_m = f_values[0]

            # Clean up temporary attribute
            if hasattr(self, '_C_initialized'):
                del self._C_initialized

        return best_x, best_f
