"""CMA-ES optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 4, cumulative step-size control."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Population size
        lam = 4
        mu = lam // 2  # 2 best individuals

        # CMA-ES parameters
        mu_eff = mu / 2.0
        c_sigma = (mu_eff + 2) / ((dim + 2)**2 + mu_eff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 / ((dim + 1.3)**2 + mu_eff)) * (1 - c_c)
        c_mu = (2 / ((dim + 1.3)**2 + mu_eff)) * min(1, mu_eff / 2 - 1)

        # Weights: log-linear weights
        w = np.log(np.arange(mu, 0, -1))
        w /= w.sum()

        # Initial state
        x = np.zeros(dim)  # start from origin (locality of f14 minimum)
        sigma = 1.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_N = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        # Evaluate initial point
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Main loop
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            ys = []
            for i in range(lam):
                z = np.random.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                ys.append(y)

            # Evaluate population
            fs = []
            for y in ys:
                if calls >= budget:
                    break
                fy = func(y)
                fs.append(fy)
                calls += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

            if len(fs) < lam:
                break

            # Sort by fitness
            idx = np.argsort(fs)
            y_sorted = [ys[i] for i in idx[:mu]]

            # Update mean
            x_new = sum(w[i] * y_sorted[i] for i in range(mu))

            # Update step size (CSA - Cumulative Step-size Adaptation)
            diff = x_new - x
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * diff))
            sigma *= np.exp(c_sigma * d_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update covariance matrix
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * diff
            rank1 = np.outer(p_c, p_c)
            rankmu = sum(w[i] * np.outer(y_sorted[i] - x, y_sorted[i] - x) for i in range(mu)) / sigma**2
            C = (1 - c_1 - c_mu) * C + c_1 * (2 + c_c - c_c) * rank1 + c_mu * rankmu

            # Eigen decomposition for sampling
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                D = np.sqrt(eigvals)
                B = eigvecs
            except np.linalg.LinAlgError:
                B = np.eye(dim)
                D = np.ones(dim)
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_sigma = np.zeros(dim)

            x = x_new

        return best_x, best_fx
# EVOLVE-BLOCK-END
