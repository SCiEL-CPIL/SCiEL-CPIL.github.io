"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for the BBOB f12 (Bent cigar) problem."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = np.inf

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / ((n + 4) ** 2 + mu_eff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 - mu_eff / (n + 1) ** 2 + 2 / (n + 1)) / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 1.3) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # State
        x = np.zeros(n)
        sigma = 1.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        calls_made = 0

        def evaluate(x_vec):
            nonlocal calls_made
            calls_made += 1
            return func(np.clip(x_vec, -5, 5))

        # Initial evaluation
        f = evaluate(x)
        best_f = f
        best_x = x.copy()

        while calls_made < budget:
            # Generate lambda_ candidates
            zs = rng.standard_normal((lambda_, n))
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = (eigvecs * eigvals) @ eigvecs.T
                L = np.linalg.cholesky(C + 1e-10 * np.eye(n))

            candidates = np.array([x + sigma * (L @ z) for z in zs])

            # Evaluate candidates
            f_values = []
            for y in candidates:
                if calls_made >= budget:
                    break
                fy = evaluate(y)
                f_values.append(fy)

            if len(f_values) < lambda_:
                break

            # Sort by fitness
            order = np.argsort(f_values)
            candidates = candidates[order]
            f_values = [f_values[i] for i in order]

            # Update best
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = candidates[0].copy()

            # Mean of best mu points
            m_new = np.zeros(n)
            for i in range(mu):
                m_new += weights[i] * candidates[i]

            # Update evolution paths
            diff = m_new - x
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (L.T @ diff)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L.T @ diff)

            # Update covariance matrix
            delta = (np.linalg.norm(ps) / chi_n > 1.4 + c_sigma) * (1 - np.exp(-2 * (np.linalg.norm(ps) / chi_n - 1) / n))
            C = (1 - c_1 - c_mu) * C + c_1 * ((1 + delta - c_1) * np.outer(pc, pc) - C)
            for i in range(mu):
                yi_minus_m = candidates[i] - m_new
                C += c_mu * weights[i] * np.outer(yi_minus_m, yi_minus_m)

            # Ensure symmetry and positive definiteness
            C = 0.5 * (C + C.T)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = (eigvecs * eigvals) @ eigvecs.T
            C = 0.5 * (C + C.T)

            # Update mean and sigma
            x = m_new
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Clip x to bounds
            x = np.clip(x, -5, 5)

            # Convergence check
            if len(f_values) >= 10:
                if np.std(f_values[:10]) < 1e-12:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
