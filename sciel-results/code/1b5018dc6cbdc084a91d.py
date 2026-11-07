"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for BBOB f20 (Schwefel) on [-5,5]^5.
    
    Uses CMA-ES with population size ~10, starting from center with
    moderate step size. CMA-ES adapts to the unknown rotation and shift
    through covariance matrix adaptation, making it robust to the
    hidden transformations.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 330s kill

        call_count = [0]

        def make_call(x):
            if call_count[0] >= budget:
                return None
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            return func(x)

        # CMA-ES parameters
        # Population size
        lam = 4 + 3 * int(np.log(dim))  # ~10 for dim=5
        # Initial parameters
        x_mean = np.zeros(dim)
        fx = make_call(x_mean)
        if fx is None:
            return x_mean, np.inf
        best_x = x_mean.copy()
        best_f = fx
        sigma = 2.0  # initial step size

        # CMA-ES weights
        mu = lam // 2  # number of selected parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # State variables
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        # Precompute C^0.5 initially as I
        C_inv_sqrt = np.eye(dim)
        C_sqrt = np.eye(dim)

        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        def update_cma_ess(x_mean, sigma, C, C_inv_sqrt, pc, ps, c_sigma, c_c, c_1, c_mu, weights, mu, mu_eff, dim, chi_n, offspring, fvals):
            """Update CMA-ES state."""
            # Sort offspring by fitness
            idx = np.argsort(fvals)
            selected = offspring[idx[:mu]]
            fvals_selected = fvals[idx[:mu]]

            # New mean
            new_mean = (weights * selected.T).T.sum(axis=0)

            # Translation
            y = (new_mean - x_mean) / sigma
            x_sigma = C_inv_sqrt @ y

            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * x_sigma
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (1 + 1))) <= (1 + 2 / (dim + 1)) * chi_n)
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y

            # Covariance matrix update
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(pc, pc) + delta_h * np.eye(dim)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                yi = C_inv_sqrt @ ((selected[i] - x_mean) / sigma)
                rank_mu += weights[i] * np.outer(yi, yi)

            C = (1 + c_1 * (delta_h - 1) + c_mu * (delta_h - 1)) * C + c_1 * rank_one + c_mu * rank_mu

            # Recompute square root
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            return new_mean, C, C_inv_sqrt, pc, ps

        # Main CMA-ES loop
        while call_count[0] < budget:
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = []
            fvals = []
            for i in range(lam):
                z = np.random.randn(dim)
                y = x_mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5.0, 5.0)
                fy = make_call(y)
                if fy is None:
                    break
                offspring.append(y)
                fvals.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(offspring) < mu:
                break

            # Update
            x_mean, C, C_inv_sqrt, pc, ps = update_cma_ess(
                x_mean, sigma, C, C_inv_sqrt, pc, ps,
                c_sigma, c_c, c_1, c_mu, weights, mu, mu_eff, dim, chi_n,
                np.array(offspring), np.array(fvals)
            )

            # Adapt sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / chi_n - 1) / 5.0)
            sigma = np.clip(sigma, 1e-8, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
