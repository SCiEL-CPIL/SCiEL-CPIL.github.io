"""Two-phase optimizer: coarse random search then CMA-ES refinement."""
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: coarse random search then CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Phase 1: Coarse random search to find a good starting point
        # Use a mix of uniform sampling and some structured points
        phase1_budget = min(100, self.budget // 5)
        x = None
        fx = np.inf
        
        # Try center first
        x0 = np.zeros(self.dim)
        fx0 = func(x0)
        x, fx = x0, fx0
        
        # Random sampling
        for i in range(phase1_budget - 1):
            if time.time() > deadline:
                break
            xi = rng.uniform(-5, 5, self.dim)
            fxi = func(xi)
            if fxi < fx:
                x, fx = xi, fxi
        
        # Phase 2: CMA-ES refinement
        remaining_budget = self.budget - (phase1_budget)
        if remaining_budget <= 0:
            return x, fx
        
        # CMA-ES implementation
        # Population size
        lam = max(4, int(4 + 3 * np.log(self.dim)))
        
        # Initialize at the best point found
        x_best = x.copy()
        f_best = fx
        
        # CMA-ES parameters
        sigma = 1.0
        # Covariance matrix
        C = np.eye(self.dim)
        # Evolution paths
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        # Learning rates
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / self.dim) / ((self.dim + 2)**2 + mu_eff))
        
        # Damping for sigma update
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        
        # Initial sigma based on problem scale
        # Start with a moderate sigma
        sigma = 1.5
        
        # CMA-ES main loop
        calls_made = phase1_budget
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Generate population
            population = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                # Sample from multivariate normal
                z = rng.standard_normal(self.dim)
                # Transform using Cholesky decomposition of C
                # For efficiency, use eigendecomposition when needed
                # Simple approach: use np.linalg.cholesky
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity if Cholesky fails
                    L = np.eye(self.dim)
                y = L @ z
                xi = x_best + sigma * y
                # Clip to box
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                population.append((fi, xi))
                calls_made += 1
            
            if not population:
                break
            
            # Sort by fitness
            population.sort(key=lambda t: t[0])
            
            # Update best
            f_best_iter, x_best_iter = population[0]
            if f_best_iter < f_best:
                f_best = f_best_iter
                x_best = x_best_iter.copy()
            
            # Update mean
            old_mean = x_best.copy()
            x_new = sum(weights[i] * population[i][1] for i in range(mu))
            
            # Update evolution path for sigma
            # Need to compute C^{-1/2} (x_new - old_mean) / sigma
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(self.dim)
            
            diff = (x_new - old_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ diff
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)
            
            # Update evolution path for covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ diff
            
            # Update covariance
            # Rank-one update
            delta_x = x_new - old_mean
            C = (1 - c_1 - c_mu) * C + c_1 * (C + np.outer(p_c, p_c) - np.eye(self.dim)) * 0 + c_1 * np.outer(delta_x, delta_x) / sigma**2
            
            # Rank-mu update
            for i in range(mu):
                y_i = (population[i][1] - old_mean) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Ensure positive semi-definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = (C + C.T) / 2
            except np.linalg.LinAlgError:
                C = np.eye(self.dim)
            
            # Update mean
            x_best = x_new
        
        return x_best, f_best
