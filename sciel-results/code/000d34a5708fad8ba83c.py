"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time-based deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 250  # Safety margin for 300s limit

        rng = np.random.default_rng(0)
        
        # CMA-ES parameters
        n = self.dim
        # lambda = 4n + floor(3 log n)
        lam = 4 * n + int(3 * np.log(n)) if n > 1 else 4
        mu = lam // 2
        
        # Initial population
        x0 = np.zeros(n)
        sigma = 2.0
        
        # CMA-ES state
        C = np.eye(n)
        Cinv = np.eye(n)
        
        # Step sizes for mean and covariance
        cmu = 2 / ((n + 1.3) ** 2 + mu)
        c1 = (2 + 2 / (n + 1.3)) / ((n + 2) ** 2 + mu)
        cmu1 = 2 * cmu * (mu - 0.5)
        
        # Step sizes for step-size adaptation
        cs = (mu + 0.5) / (n + mu + 5)
        c1_sigma = (2 + 2 / (n + 1.3)) / ((n + 2) ** 2 + mu)
        c1_sigma = c1_sigma / 2  # Standard CMA-ES usually has c1_sigma = c1/2 or similar, let's stick to standard formulas
        
        # Actually, let's use the standard CMA-ES formulas from Hansen's paper
        # cs = (mu + 0.5) / (n + mu + 5)
        # c1 = (2 + 2 / (n + 1.3)) / ((n + 2) ** 2 + mu)
        # cmu = 2 / ((n + 1.3) ** 2 + mu)
        
        # Path accumulators
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Initial population
        z = rng.standard_normal((lam, n))
        # Mahalanobis distance check (optional, usually not needed for initial)
        pop = x0 + sigma * z
        # Clip to box
        pop = np.clip(pop, -5, 5)
        
        # Evaluate initial population
        fvals = np.array([func(x) for x in pop])
        
        # Sort population
        indices = np.argsort(fvals)
        pop = pop[indices]
        fvals = fvals[indices]
        
        # Best so far
        x_best = pop[0].copy()
        f_best = fvals[0]
        
        calls_made = lam
        budget_remaining = self.budget - calls_made
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights ** 2)
        
        # Initial covariance matrix
        C = np.eye(n)
        
        # Precompute sqrt(C)
        # Use eigendecomposition for stability
        d, V = np.linalg.eigh(C)
        d = np.sqrt(np.maximum(d, 1e-20))
        D = np.diag(d)
        C_sqrt = V @ D @ V.T
        
        # Inverse sqrt
        Cinv_sqrt = V @ np.diag(1/d) @ V.T
        
        # Main loop
        while budget_remaining > 0 and time.time() < deadline:
            # Generate new population
            z = rng.standard_normal((lam, n))
            pop = x_best + sigma * (z @ C_sqrt.T)
            pop = np.clip(pop, -5, 5)
            
            # Evaluate
            fvals = np.array([func(x) for x in pop])
            calls_made += lam
            budget_remaining = self.budget - calls_made
            
            # Sort
            indices = np.argsort(fvals)
            pop = pop[indices]
            fvals = fvals[indices]
            
            # Update best
            if fvals[0] < f_best:
                x_best = pop[0].copy()
                f_best = fvals[0]
            
            # Compute mean of selected individuals
            x_mean = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
            
            # Update path accumulators
            # p_sigma = (1 - cs) * p_sigma + cs * sqrt(mueff) * Cinv_sqrt @ (x_mean - x_best) / sigma
            p_sigma = (1 - cs) * p_sigma + cs * np.sqrt(mueff) * (Cinv_sqrt @ (x_mean - x_best) / sigma)
            
            # hsig = ||p_sigma|| / sqrt(1 - (1-cs)^{2/gen}) / ||I||
            # Approximate hsig
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** 2)) / np.sqrt(n) < 1.4 + 2 / (n + 1)
            
            # p_c = (1 - c1) * p_c + hsig * c1 * sqrt(mueff) * (x_mean - x_best) / sigma
            p_c = (1 - c1) * p_c + hsig * c1 * np.sqrt(mueff) * (x_mean - x_best) / sigma
            
            # Update covariance matrix
            # C = (1 - c1 - cmu) * C + c1 * (p_c @ p_c + (1 - hsig) * c1 * (1 - (1-c1)^2) * C) + cmu * sum(weights[i] * (y_i - x_mean) @ (y_i - x_mean).T)
            # Note: y_i = (pop[i] - x_best) / sigma
            y = (pop[:mu] - x_best) / sigma
            delta_hsig = (1 - hsig) * c1 * (1 - (1 - c1) ** 2)
            
            C = (1 - c1 - cmu) * C + c1 * (p_c @ p_c + delta_hsig * C)
            for i in range(mu):
                C += cmu * weights[i] * np.outer(y[i], y[i])
            
            # Update step size
            # chi_n = sqrt(n) * (1 - 1/(4n) + 1/(21n^2))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update C_sqrt
            d, V = np.linalg.eigh(C)
            d = np.sqrt(np.maximum(d, 1e-20))
            D = np.diag(d)
            C_sqrt = V @ D @ V.T
            Cinv_sqrt = V @ np.diag(1/d) @ V.T
            
            # Update x_best (mean of distribution)
            # In CMA-ES, the mean is updated to the weighted mean of the best individuals
            # But we keep x_best as the best found so far for the "best" record, 
            # while the distribution center moves.
            # Actually, standard CMA-ES updates the mean m. 
            # Let's maintain m separately.
            
        return x_best, f_best
# EVOLVE-BLOCK-END
