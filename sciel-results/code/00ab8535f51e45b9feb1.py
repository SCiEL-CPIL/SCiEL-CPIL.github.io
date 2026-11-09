"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style f17 (Schaffers F7)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, leave 20s margin
        
        rng = np.random.default_rng(42)
        best_x = None
        best_fx = np.inf
        calls_made = 0
        
        def call_func(x):
            nonlocal calls_made
            fx = func(x)
            calls_made += 1
            return fx
        
        # Phase 1: Diverse initial sampling
        n_initial = min(15, self.budget)
        for i in range(n_initial):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_fx = call_func(best_x)
        
        # Phase 2: CMA-ES
        remaining_budget = self.budget - calls_made
        if remaining_budget < 20:
            return best_x, best_fx
        
        # CMA-ES parameters
        n = self.dim
        # Population size: lambda = 4 + floor(3 * log(n))
        lam = max(4, 4 + int(3 * np.log(n)))
        # Initial sigma: 1.0
        sigma = 1.0
        # Mean
        mean = best_x.copy()
        # Covariance matrix
        C = np.eye(n)
        # Eigenvalues and eigenvectors of C
        eigval = np.ones(n)
        eigvec = np.eye(n)
        # Step-size adaptation
        cs = 1 / (np.sqrt(lam / 2) + n)  # damping for step-size adaptation
        ds = 1 + 2 * max(0, (n - 1) / (lam - 2)) + cs  # damping for sigma
        # Covariance matrix adaptation
        cmu = 2 / ((n + 1)**2 + lam)  # weight for covariance adaptation
        c1 = (2 + 2 / (n + 1)) / ((n + 2)**2 + lam)  # base learning rate
        c_sigma = (cs + 2 * cmu) / (n + 4)  # learning rate for p_sigma
        # Path accumulators
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        # Weights for recombination
        mu = lam // 2  # number of best solutions used for recombination
        weights = np.log(np.arange(lam, 0, -1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights[:mu]**2)  # effective population size
        # Update c1 and cmu with mu_eff
        c1 = (2 + 2 / (n + 1)) / ((n + 2)**2 + lam)
        cmu = min(2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2)**2 + lam), 1 - c1)
        # chi_n: expected norm of N(0, I)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Main CMA-ES loop
        while calls_made < self.budget and time.time() < deadline:
            if calls_made + lam > self.budget:
                break
            
            # Generate population
            offspring = np.zeros((lam, n))
            fitness = np.zeros(lam)
            
            for i in range(lam):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                # Sample from N(mean, sigma^2 * C)
                z = rng.standard_normal(n)
                # Transform using eigendecomposition
                y = eigvec @ (np.sqrt(eigval) * z)
                x_i = mean + sigma * y
                x_i = np.clip(x_i, -5, 5)
                fitness[i] = call_func(x_i)
                offspring[i] = x_i
            
            if calls_made < n_initial + lam:
                # First generation: update best
                for i in range(lam):
                    if fitness[i] < best_fx:
                        best_fx = fitness[i]
                        best_x = offspring[i].copy()
            
            # Sort by fitness
            idx = np.argsort(fitness)
            offspring_sorted = offspring[idx]
            fitness_sorted = fitness[idx]
            
            # Update best
            for i in range(lam):
                if fitness_sorted[i] < best_fx:
                    best_fx = fitness_sorted[i]
                    best_x = offspring_sorted[i].copy()
            
            # Weighted recombination
            y_w = np.zeros(n)
            for i in range(mu):
                y_w += weights[i] * (offspring_sorted[i] - mean) / sigma
            
            # Update step-size adaptation
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (eigvec @ (np.sqrt(eigval) * (y_w / np.linalg.norm(y_w) + 1e-15)))
            
            # Hessian penalty
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - n_initial + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0
            
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)
            
            # Update covariance matrix adaptation
            p_c = (1 - c1) * p_c + h_sigma * np.sqrt(c1 * (2 - c1) * mu_eff) * (eigvec @ (np.sqrt(eigval) * y_w))
            
            delta_h = (1 - h_sigma) * c1 * (2 - c1)
            C = (1 - c1 - cmu) * C + c1 * (p_c @ p_c + delta_h * C)
            for i in range(mu):
                y_i = (offspring_sorted[i] - mean) / sigma
                C += cmu * weights[i] * (y_i @ y_i)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Eigen-decomposition (periodic to save time)
            if (calls_made - n_initial) % 10 < lam or calls_made < n_initial + lam:
                eigval, eigvec = np.linalg.eigh(C)
                eigval = np.maximum(eigval, 1e-12)
            
            # Update mean
            mean = mean + np.sum(weights[:mu, np.newaxis] * offspring_sorted[:mu], axis=0)
            mean = np.clip(mean, -5, 5)
            
            # Convergence check: if sigma is very small or no improvement for many generations
            if sigma < 1e-10:
                break
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
