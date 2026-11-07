"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget

        # Initial point: center of the box (minimizer is in [-4,4]^5)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        count = 1

        # Initial step size
        sigma = 1.0

        # CMA-ES parameters
        pop_size = 4 + int(3 * np.log(dim))
        pop_size = min(pop_size, budget - 1)

        # Covariance matrix (start with identity)
        C = np.eye(dim)
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Adaptation rates
        cs = (pop_size + 2) / (dim + 4)
        cmu = 2 / ((dim + 1.3) ** 2 + pop_size)
        c1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        c_sigma = (pop_size + 1.7) / (dim + 18)

        # Eigen decomposition cache
        eigvals, eigvecs = np.linalg.eigh(C)

        while count < budget:
            # Check time
            if time.time() > deadline - 5:
                break

            # Generate population
            ys = []
            fvals = []
            for i in range(pop_size):
                # Sample from multivariate normal with covariance C
                z = np.random.randn(dim)
                y = x + sigma * (eigvecs @ (eigvals ** 0.5 * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                count += 1
                ys.append(y)
                fvals.append(fy)

                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

                # Time check within population
                if time.time() > deadline - 10:
                    break

            # Sort by fitness
            order = np.argsort(fvals)
            ys_sorted = [ys[i] for i in order]
            fvals_sorted = [fvals[i] for i in order]

            # Update mean (weighted by rank)
            weights = np.log(pop_size / 2 + 1) - np.log(np.arange(1, pop_size + 1))
            weights /= weights.sum()
            mu_eff = (weights ** 2).sum()
            
            # Only use top half for mean update
            top_n = pop_size // 2
            w_top = weights[:top_n]
            w_top /= w_top.sum()
            mu_eff_top = (w_top ** 2).sum()

            new_x = sum(w_top[i] * ys_sorted[i] for i in range(top_n))
            
            # Update sigma
            sigma_old = sigma
            sigma *= np.exp(c_sigma / mu_eff_top * (np.linalg.norm(new_x - x) / sigma_old - 1))
            sigma = max(sigma, 1e-10)

            # Update evolution paths
            d = new_x - x
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff_top) * d / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) / np.sqrt(mu_eff_top) <
                    (1 + 2 / (dim + 1)) * np.sqrt(1 - (1 - cs) ** 2))
            
            x_old = x
            x = new_x
            
            # Update covariance
            if hsig:
                pc = (1 - cmu) * pc + np.sqrt(cmu * (2 - cmu) * mu_eff_top) * d / sigma
            else:
                pc = (1 - cmu) * pc
            
            rank_mu = np.zeros((dim, dim))
            for i in range(top_n):
                yi = (ys_sorted[i] - x_old) / sigma
                rank_mu += w_top[i] * np.outer(yi, yi)
            
            C = (1 + c1 * (2 - c1) + cmu - c1) * C + c1 * (np.outer(pc, pc) - C) + cmu * rank_mu

            # Re-eigen decompose periodically (every 10 generations)
            if count % (10 * pop_size) < pop_size:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive eigenvalues
                eigvals = np.maximum(eigvals, 1e-12)

        return best_x, best_f
# EVOLVE-BLOCK-END
