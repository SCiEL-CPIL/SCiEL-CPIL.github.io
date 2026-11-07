"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f3 (Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # CMA-ES Parameters
        # Population size
        lambda_ = 4 * dim + 2  # 22 for dim=5
        # Weights
        weights = np.log(np.arange(1, lambda_ + 1))
        weights = weights / np.sum(weights)
        mu = int(np.floor(lambda_ / 2))  # Half of the population
        w = weights[:mu]
        w = w / np.sum(w)
        mu_eff = 1 / np.sum(w ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initialization
        x0 = np.zeros(dim)
        sigma = 2.0  # Initial step size
        m = x0.copy()
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Cholesky decomposition of C
        try:
            B = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            B = np.eye(dim)
        D = np.sqrt(np.diag(B))
        inv_D = 1.0 / D
        inv_B = np.linalg.inv(B)
        
        hs = np.sqrt((1 + c_c) / (1 - c_c)) + 2

        best_x = x0.copy()
        best_f = func(best_x)
        calls = 1

        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            ys = []
            xs = []
            fs = []
            
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                z = np.random.standard_normal(dim)
                y = B @ (inv_D * z)
                x = m + sigma * y
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls += 1
                ys.append(y)
                xs.append(x)
                fs.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            if len(xs) < lambda_:
                break

            ys = np.array(ys)
            xs = np.array(xs)
            fs = np.array(fs)

            # Sort by fitness
            order = np.argsort(fs)
            xs = xs[order]
            ys = ys[order]
            fs = fs[order]

            # Update mean
            m_new = m + sigma * np.sum(w[:, None] * (xs[:mu] - m) / sigma, axis=0)
            # Actually, the mean update is: m = sum(w_i * x_i)
            m = np.sum(w[:, None] * xs[:mu], axis=0)
            
            # Update covariance matrix
            ysig = (xs[:mu] - m_new) / sigma  # This is approximate, usually we use the selected individuals relative to the old mean
            # Correct update:
            ysig = (xs[:mu] - m) / sigma
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - (1 - c_sigma) ** 2) * np.eye(dim)) + \
                c_mu * np.sum(w[:, None, None] * np.outer(ysig, ysig), axis=0)
            
            # Update path lengths
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * B @ (inv_D * (m - m_new) / sigma)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * B @ (inv_D * (m - m_new) / sigma)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / hs * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            
            # Re-estimate B and D
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, add small diagonal
                C += 1e-6 * np.eye(dim)
                try:
                    B = np.linalg.cholesky(C)
                except:
                    break
            D = np.sqrt(np.diag(B))
            inv_D = 1.0 / D
            inv_B = np.linalg.inv(B)

            if calls >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
