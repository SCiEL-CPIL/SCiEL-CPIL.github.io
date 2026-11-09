"""Optimized program for BBOB f6 using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for the BBOB f6 (Attractive Sector) problem."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # CMA-ES parameters
        # Population size lambda
        self.lambda_ = 4 + int(3 * np.log(dim))
        # Step size sigma
        self.sigma = 2.0
        # Initial mean x
        # The minimizer is in [-4, 4]^5. Starting at 0 is a good bet.
        self.x_mean = np.zeros(dim)
        # Covariance matrix C
        self.C = np.eye(dim)
        # Inverse square root of C (for sampling)
        # We can use Cholesky or eigendecomposition. For dim=5, Cholesky is fine.
        # But standard CMA-ES uses C^{1/2}. We maintain C and compute C^{1/2} via Cholesky each step? 
        # Or maintain C^{1/2} and update it? 
        # For simplicity and robustness in a short budget, we will compute C^{1/2} via Cholesky in each generation. 
        # Dim=5 is small, so this is cheap.
        
        # Evolution paths
        self.p_sigma = np.zeros(dim)
        self.p_c = np.zeros(dim)
        
        # Learning rates
        self.mu = self.lambda_ // 2
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)
        self.mu_eff = 1.0 / np.sum(self.weights**2)
        
        self.c_sigma = (self.mu_eff + 2) / (dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, (self.mu_eff - 1) / (dim + 1)) + self.c_sigma
        self.c_c = (4 + self.mu_eff / dim) / (dim + 4 + 2 * self.mu_eff / dim)
        self.c_1 = 2 / ((dim + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((dim + 2)**2 + self.mu_eff))
        
        self.chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s safety margin
        
        x = self.x_mean.copy()
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        
        # Pre-allocate storage for population
        # We will generate the whole population, evaluate, then update
        
        while calls < self.budget:
            # Check time
            if time.time() > deadline - 1.0:
                break
            
            # Generate population
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(self.C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes singular (rare)
                L = np.eye(self.dim)
            
            # Sample standard normal vectors
            # Shape: (lambda, dim)
            normal_samples = np.random.default_rng(0).standard_normal((self.lambda_, self.dim))
            # Transform to CMA-ES distribution
            # y = sigma * L @ z
            # Note: L is lower triangular. z @ L.T is equivalent to L @ z if we handle shapes correctly.
            # Actually, C^{1/2} = L. So x = x_mean + sigma * (z @ L.T)
            # Let's verify: Cov(z @ L.T) = L Cov(z) L.T = L I L.T = L L.T = C. Correct.
            
            z = np.random.default_rng(0).standard_normal((self.lambda_, self.dim))
            y = self.sigma * (z @ L.T)
            
            # Generate candidates
            candidates = x + y
            # Clip to box [-5, 5]
            candidates = np.clip(candidates, -5, 5)
            
            # Evaluate
            fvals = np.array([func(c) for c in candidates])
            calls += self.lambda_
            
            # Update best
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < best_f:
                best_f = fvals[idx_min]
                best_x = candidates[idx_min].copy()
            
            # Sort candidates by fitness
            order = np.argsort(fvals)
            y_sorted = y[order]
            f_sorted = fvals[order]
            
            # Weighted mean of best mu points
            y_w = np.dot(self.weights[:self.mu], y_sorted[:self.mu])
            
            # Update mean
            x = x + self.sigma * y_w
            
            # Update p_sigma
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1 - cs)^{2*(gen+1)}) < (1.4 + 2/(dim+1)) * chi_n
            # We don't track gen explicitly here, but we can approximate or just use the standard update
            # For simplicity and robustness in a short run, we can skip the h_sigma test or implement it simply.
            # Let's implement the standard h_sigma test.
            # We need to track the number of generations.
            # Let's add a gen counter.
            
            # Actually, let's keep it simple. The h_sigma test prevents premature convergence of sigma.
            # We'll track gen.
            
            # Update p_c
            pc_old = self.p_c.copy()
            self.p_c = (1 - self.c_c) * self.p_c + self.c_c * (y_w / self.sigma)
            h_c = np.linalg.norm(self.p_c) / np.sqrt(1 - (1 - self.c_c)**2) < (1.4 + 2 / (self.dim + 1)) * self.chi_n
            
            # Update p_sigma
            if h_c:
                self.p_sigma = (1 - self.c_sigma) * self.p_sigma + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * (y_w / self.sigma)
            else:
                self.p_sigma = (1 - self.c_sigma) * self.p_sigma
            
            # Update sigma
            # h_sigma logic
            # We need the norm of p_sigma
            # The threshold is (1.4 + 2/(dim+1)) * chi_n
            # The denominator is sqrt(1 - (1-cs)^{2*(gen+1)})
            # Since we don't have gen, we can approximate (1-cs)^{2*(gen+1)} as 0 for large gen, so denom ~ 1.
            # For early gen, it's close to 1.
            # Let's assume denom ~ 1 for simplicity, which is safe.
            h_sigma = np.linalg.norm(self.p_sigma) / self.chi_n < (1.4 + 2 / (self.dim + 1))
            
            if h_sigma:
                self.sigma *= np.exp(self.c_sigma / self.d_sigma * (np.linalg.norm(self.p_sigma) / self.chi_n - 1))
            
            # Update Covariance Matrix C
            # Delta H term
            delta_h = (1 - h_c) * self.c_c * (2 - self.c_c) * np.outer(self.p_c, self.p_c)
            
            # Rank-mu update
            # Sum of outer products of weighted y's
            # y_sorted[:mu] are the best ones
            # We need to scale them by 1/sigma
            y_mu = y_sorted[:self.mu] / self.sigma
            rank_mu_update = np.dot(self.weights[:self.mu, np.newaxis], np.outer(y_mu, np.ones((self.mu, self.dim)))).sum(axis=0)
            # Wait, the formula is sum w_i (y_i y_i^T)
            # y_mu is (mu, dim). We want (dim, dim).
            # rank_mu_update = sum_{i=0}^{mu-1} w_i * y_mu[i] * y_mu[i].T
            rank_mu_update = np.zeros((self.dim, self.dim))
            for i in range(self.mu):
                rank_mu_update += self.weights[i] * np.outer(y_mu[i], y_mu[i])
            
            self.C = (1 + self.c_1 * (2 - self.c_1) - self.c_mu) * self.C + self.c_mu * rank_mu_update + delta_h
            
            # Symmetrize C to avoid numerical drift
            self.C = (self.C + self.C.T) / 2.0
            
            # Ensure C is positive definite (add small identity if needed)
            # For dim=5, this should be fine.
            
        return best_x, best_f

# EVOLVE-BLOCK-END
