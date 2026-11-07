import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f12 (Bent cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        eval_count = 0
        best_x = np.zeros(d)
        best_f = float('inf')

        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5, 5)
            f = func(x)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(n))
        lam = max(4, int(4 + 3 * np.log(d)))
        # Initial sigma: 2.0 (standard for BBOB)
        sigma = 2.0
        # Initial mean: center of the box
        mean = np.zeros(d)
        # Covariance matrix: identity
        C = np.eye(d)
        # Eigen decomposition for efficient sampling
        eigvals, eigvecs = np.linalg.eigh(C)
        # Step size control parameters
        c_sigma = (mu + 0.3) / (d + 4)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (d + 1)) - 1) + c_sigma
        # Mu: number of recombination weights
        mu = lam // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        # Covariance adaptation parameters
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        # Path parameters
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # Initial evaluations: evaluate the center and a few random points to get a better start
        # Actually, let's just start CMA-ES from the center
        # But first, do a few random evaluations to get a rough sense of the landscape
        rng = np.random.default_rng(42)
        
        # Initial phase: random search to find a good starting point
        initial_budget = min(20, budget // 5)
        for _ in range(initial_budget):
            if eval_count >= budget or time.time() > deadline:
                break
            x_rand = rng.uniform(-4, 4, d)
            f_rand = eval_func(x_rand)
            if f_rand < best_f:
                mean = best_x.copy()

        # CMA-ES main loop
        while eval_count < budget and time.time() < deadline:
            # Generate lambda offspring
            offspring = np.zeros((lam, d))
            f_offspring = np.zeros(lam)
            
            # Update eigen decomposition if needed (do it every iteration for simplicity, or less frequently)
            # For efficiency, we can update less frequently, but for small d it's fine
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            
            for i in range(lam):
                if eval_count >= budget or time.time() > deadline:
                    break
                # Sample y ~ N(0, C)
                z = rng.standard_normal(d)
                y = eigvecs @ (np.sqrt(eigvals) * z)
                x = mean + sigma * y
                x = np.clip(x, -5, 5)
                offspring[i] = x
                f_offspring[i] = eval_func(x)
            
            if eval_count >= budget or time.time() > deadline:
                break
            
            # Sort offspring by fitness
            order = np.argsort(f_offspring)
            sorted_f = f_offspring[order]
            sorted_x = offspring[order]
            
            # Update mean (recombination)
            new_mean = np.zeros(d)
            for i in range(mu):
                new_mean += weights[i] * sorted_x[i]
            
            # Update sigma (path adaptation)
            # Check if the mean moved
            if np.linalg.norm(new_mean - mean) > 1e-16:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (eigvecs.T @ (eigvecs @ (new_mean - mean) / sigma))
            else:
                p_sigma = (1 - c_sigma) * p_sigma
            
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 100.0)
            
            # Update covariance matrix
            # Path for rank-one update
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma
            
            # Rank-one update
            delta_h = (1.04 + 2 / (d + 1)) * np.linalg.norm(p_c) / chi_n < 1.4 + 2 / (d + 1)
            h = 1.0 if delta_h else 0.0
            
            C = (1 + c_c * (2 - c_c - c_1) * h * (np.outer(p_c, p_c) - C) + c_1) * C
            # Rank-mu update
            for i in range(mu):
                y_i = (sorted_x[i] - mean) / sigma
                C += c_mu * np.outer(y_i, y_i)
            
            # Symmetrize and ensure positive semi-definite
            C = (C + C.T) / 2
            # Add small regularization for numerical stability
            C += 1e-12 * np.eye(d)
            
            # Update mean
            mean = new_mean
            
            # Update best
            if sorted_f[0] < best_f:
                best_f = sorted_f[0]
                best_x = sorted_x[0].copy()

            # Check for convergence
            if sigma < 1e-10:
                break
            
            # If we've made very little progress, consider restarting
            # For now, just continue

        return best_x, best_f
