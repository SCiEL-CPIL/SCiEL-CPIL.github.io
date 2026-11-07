"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 300s limit
        
        # CMA-ES parameters
        # For 5D, lambda (population size) = 4 + floor(3 * log(5)) = 4 + 4 = 8
        # Using lambda=10 for slightly better exploration in early stages
        lambda_ = 10
        mu = max(1, lambda_ // 2)  # Number of recombination points
        # Weights for recombination (log-sum normalization)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Initial parameters
        # Start at origin, which is a reasonable guess for BBOB functions
        x_mean = np.zeros(self.dim)
        sigma = 2.0  # Initial step size, covering a good portion of [-5, 5]
        
        # CMA parameters
        # Standard CMA-ES adaptation rates
        c_sigma = (mueff + 2) / (self.dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / self.dim) / (self.dim + 4 + 2 * mueff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff - 2 + 1/mueff) / ((self.dim + 2)**2 + mueff)))
        
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        
        # Precompute sqrt of C for sampling
        # We will update this periodically or use Cholesky
        # For simplicity and speed in 5D, we can compute Cholesky each generation
        # or use SVD. Cholesky is faster for small dim.
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        call_count = 1
        
        # Track history for AOCC (though evaluator does it, we need to return best)
        # The evaluator records every call. We just need to make good calls.
        
        while call_count + lambda_ <= self.budget:
            if time.time() > deadline:
                break
                
            # Generate population
            # Compute Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset or add jitter
                C = np.eye(self.dim)
                L = np.eye(self.dim)
            
            # Sample lambda points
            z = np.random.randn(lambda_, self.dim)
            # Transform to x-space: x = x_mean + sigma * L @ z
            pop = x_mean[None, :] + sigma * (z @ L.T)
            
            # Evaluate population
            f_pop = np.zeros(lambda_)
            for i in range(lambda_):
                if call_count >= self.budget or time.time() > deadline:
                    break
                # Clip to box [-5, 5]
                x_i = np.clip(pop[i], -5, 5)
                try:
                    f_pop[i] = func(x_i)
                except Exception:
                    f_pop[i] = np.inf
                call_count += 1
            
            if call_count < self.budget + lambda_:
                # If we broke early due to deadline/budget, we might have inf values
                # Ensure we don't crash on sorting
                pass
            
            # Sort by fitness
            idx = np.argsort(f_pop)
            # Take top mu
            idx_mu = idx[:mu]
            
            # Update best
            if f_pop[idx[0]] < best_f:
                best_f = f_pop[idx[0]]
                best_x = np.clip(pop[idx[0]], -5, 5)
            
            # Recombination
            new_x_mean = np.sum(weights[:, None] * pop[idx_mu], axis=0)
            
            # Update covariance matrix
            # pc update
            pc = (1 - c_c) * pc + c_c * (new_x_mean - x_mean) / sigma
            ccov1pc = (1 - c_c) * (1 + 2 * d_sigma * c_c / (1 - c_c))
            
            # ps update
            # Need inverse sqrt of C
            # Use cholesky inverse: C^-0.5 = L^-T @ L^-1
            # Since C = L @ L.T, C^-0.5 = L^-T @ L^-1
            # z_mean = C^-0.5 @ (new_x_mean - x_mean) / sigma
            # Solve L.T @ y = (new_x_mean - x_mean) / sigma
            # Then z_mean = L^-1 @ y
            try:
                diff = (new_x_mean - x_mean) / sigma
                # Solve L.T @ y = diff
                y = np.linalg.solve(L.T, diff)
                # Solve L @ z_mean = y
                z_mean = np.linalg.solve(L, y)
            except np.linalg.LinAlgError:
                z_mean = np.zeros(self.dim)
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * z_mean
            
            # Check if ps is too large (sigma adaptation)
            if np.dot(ps, ps) / (1 - (1 - c_sigma)**(2 * (call_count + lambda_))) < (1.4 + 2 / (self.dim + 1)) * self.dim:
                sigma *= np.exp(c_sigma / d_sigma * (np.dot(ps, ps) / self.dim - 1))
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (c_c * (2 - c_c)) * (np.dot(pc, pc) / self.dim - 1) * C)
            #    + c_mu * sum(weights[i] * (pop[i] - x_mean) @ (pop[i] - x_mean).T)
            
            # Term 1: C update
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + (c_c * (2 - c_c)) * (np.dot(pc, pc) / self.dim - 1) * np.eye(self.dim))
            
            # Term 2: Rank-mu update
            # We need to compute the weighted sum of outer products of the top mu points
            # relative to the OLD x_mean (standard CMA-ES uses old mean for this term in some variants, 
            # but usually it's (x_i - x_mean_old))
            # Let's stick to standard: (x_i - x_mean) where x_mean is the old mean
            for i in range(mu):
                d = pop[idx_mu[i]] - x_mean
                C += c_mu * weights[i] * np.outer(d, d)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = new_x_mean
            
            # Optional: Reset C if it becomes ill-conditioned
            # For 5D, this is rarely an issue with standard params
            
        return best_x, best_f

# EVOLVE-BLOCK-END
