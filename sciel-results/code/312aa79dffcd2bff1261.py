"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for continuous black-box problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        call_count = [0]

        def wrapped(x):
            if call_count[0] >= budget:
                return float('inf')
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            call_count[0] += 1
            if not np.isfinite(val):
                return 1e30
            return float(val)

        # CMA-ES Parameters
        lam = 7
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))

        x_mean = np.zeros(dim)
        sigma = 1.5
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)

        best_x = x_mean.copy()
        best_f = wrapped(best_x)

        gen = 0
        while call_count[0] < budget:
            if budget - call_count[0] < lam:
                break

            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)

            pop_x = []
            pop_f = []

            for i in range(lam):
                z = rng.standard_normal(dim)
                x_new = x_mean + sigma * (L @ z)
                x_new = np.clip(x_new, -5.0, 5.0)

                f_new = wrapped(x_new)
                pop_x.append(x_new)
                pop_f.append(f_new)

                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()

            if call_count[0] >= budget:
                break

            idx = np.argsort(pop_f)
            pop_x = np.array(pop_x)[idx]
            pop_f = np.array(pop_f)[idx]

            x_new_mean = np.sum(weights[:, None] * pop_x[:mu], axis=0)

            try:
                # L_inv.T @ v is equivalent to solving L.T @ w = v
                step_vector = np.linalg.solve(L.T, x_new_mean - x_mean)
            except:
                step_vector = x_new_mean - x_mean

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * step_vector / sigma

            decay = (1 - c_sigma) ** (2 * (gen + 1))
            denom = np.sqrt(1 - decay) if 1 - decay > 0 else 1.0
            threshold = (1.4 + 2 / (dim + 1)) * chi_n
            h_sigma = 1 if (np.linalg.norm(ps) / denom < threshold) else 0

            pc = (1 - c_c) * pc + h_sigma * c_c * (x_new_mean - x_mean) / sigma

            C = (1 + c_1 * h_sigma * (2 - c_c) - c_1 - c_mu) * C \
                + c_1 * np.outer(pc, pc) \
                + c_mu * np.sum(weights[:, None, None] * np.outer(pop_x[:mu] - x_mean, pop_x[:mu] - x_mean) / sigma**2, axis=0)

            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            x_mean = x_new_mean
            gen += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
