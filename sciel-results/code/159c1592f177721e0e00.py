# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f7 (Step ellipsoidal) on [-5,5]^d."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Reset state
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

        def count_eval(x):
            self._call_count += 1
            val = func(x)
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(dim))
        lam = 4 + int(3 * np.log(dim))  # 4 + 3*1.61 = 8 for dim=5
        mu = lam // 2  # number of recombination weights
        
        # Recombination weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Step size sigma
        sigma = 2.0  # initial step size
        
        # Mean vector
        mean = rng.uniform(-2, 2, dim)
        
        # Covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Update rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_cov = (2 + 2 * mu_eff / dim) / (dim + 2 + mu_eff / dim)
        
        # Eigenvalue decay
        eigenvalues = np.ones(dim)
        
        # Counter for stagnation
        no_improve = 0
        stagnation_limit = 15 * lam
        
        # Initial evaluation
        fvals = np.empty(lam)
        for i in range(lam):
            if self._call_count >= budget or time.time() >= deadline:
                break
            fvals[i] = count_eval(mean)
        
        if len(fvals) < lam:
            fvals = fvals[:len(fvals)]
            lam = len(fvals)
            mu = max(1, lam // 2)
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights = weights / np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Track best
        best_idx = np.argmin(fvals)
        best_x = mean.copy()
        best_f = fvals[best_idx]
        
        # Main CMA-ES loop
        while self._call_count + lam <= budget and time.time() < deadline:
            # Generate lambda samples
            samples = np.empty((lam, dim))
            for i in range(lam):
                if self._call_count >= budget or time.time() >= deadline:
                    break
                z = rng.standard_normal(dim)
                # Use Cholesky decomposition of C for sampling
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                samples[i] = mean + sigma * (L @ z)
                samples[i] = np.clip(samples[i], -5, 5)
            
            if len(samples) < lam:
                samples = samples[:len(samples)]
                lam = len(samples)
            
            # Evaluate samples
            fvals = np.empty(lam)
            for i in range(lam):
                if self._call_count >= budget or time.time() >= deadline:
                    break
                fvals[i] = count_eval(samples[i])
            
            if len(fvals) < lam:
                fvals = fvals[:len(fvals)]
                lam = len(fvals)
            
            # Sort by fitness
            order = np.argsort(fvals)
            samples = samples[order]
            fvals = fvals[order]
            
            # Update best
            if fvals[0] < best_f - 1e-12:
                best_f = fvals[0]
                best_x = samples[0].copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # Update mean (weighted recombination)
            if mu > 0 and mu <= lam:
                old_mean = mean.copy()
                mean = np.sum(weights[:mu, None] * samples[:mu], axis=0)
                
                # Update sigma (CSA)
                z = (mean - old_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z
                if np.linalg.norm(p_sigma) < (1.4 + 1/dim) * (1 - c_sigma) * np.sqrt(dim):
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(dim) - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 50.0)
                
                # Update covariance matrix
                y = (mean - old_mean) / sigma
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y
                
                # Rank-one and rank-mu updates
                C = (1 + c_cov * (c_c + 2 - c_c) - c_cov) * C
                C += c_cov * (np.outer(y, y) - C)
                
                # Rank-mu update
                C += c_cov * (1 - (c_c + 2 - c_c)) * sum(
                    weights[i] * np.outer(samples[i] - old_mean, samples[i] - old_mean) / sigma**2
                    for i in range(mu)
                )
                
                # Symmetrize and ensure positive semi-definite
                C = (C + C.T) / 2
                # Regularize
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-12:
                    C += (1e-10 - np.min(eigvals)) * np.eye(dim)
            
            # Check for stagnation
            if no_improve > stagnation_limit:
                # Restart: re-seed around best solution
                sigma = max(0.5, sigma * 0.5)
                mean = best_x + rng.standard_normal(dim) * sigma * 0.5
                mean = np.clip(mean, -5, 5)
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improve = 0
        
        # Final local refinement using coordinate descent on best solution
        if self._call_count < budget - dim and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            
            for step in step_sizes:
                if self._call_count >= budget or time.time() >= deadline:
                    break
                improved = True
                while improved:
                    improved = False
                    for d in range(dim):
                        if self._call_count >= budget or time.time() >= deadline:
                            break
                        # Try positive step
                        x_try = x_local.copy()
                        x_try[d] = min(5.0, x_try[d] + step)
                        f_try = count_eval(x_try)
                        if f_try < f_local - 1e-12:
                            x_local = x_try
                            f_local = f_try
                            improved = True
                        # Try negative step
                        if self._call_count >= budget or time.time() >= deadline:
                            break
                        x_try = x_local.copy()
                        x_try[d] = max(-5.0, x_try[d] - step)
                        f_try = count_eval(x_try)
                        if f_try < f_local - 1e-12:
                            x_local = x_try
                            f_local = f_try
                            improved = True
                
                # Update best if improved
                if f_local < best_f:
                    best_f = f_local
                    best_x = x_local.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
