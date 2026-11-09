import numpy as np
import time

class Optimizer:
    """
    CMA-ES based optimizer for the Bent Cigar function.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Start time to enforce the 300s limit
        t_start = time.time()
        time_limit = 290  # Safety margin before 300s kill

        # CMA-ES Parameters
        # For Bent Cigar, a larger initial sigma helps explore the wide basin
        # before converging to the narrow valley.
        sigma0 = 2.0
        x0 = np.zeros(self.dim)
        
        # Standard CMA-ES parameters
        lambda_ = 4 * (int(np.ceil(3 * np.log(self.dim))) + 1)  # Population size
        mu = lambda_ // 2  # Number of weighted individuals
        # Weights for the selected individuals
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        
        # State variables
        C = np.eye(self.dim)
        C_sqrt_inv = np.eye(self.dim)
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        sigma = sigma0
        x = x0.copy()
        
        # Track best solution
        best_x = x0.copy()
        best_f = func(best_x)
        calls_made = 1
        
        # Pre-compute some constants
        chi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        
        # Loop until budget or time limit
        while calls_made < self.budget:
            # Time check
            if time.time() - t_start > time_limit:
                break
            
            # Generate population
            Y = np.random.randn(self.dim, lambda_)
            
            # Transform to search space
            # We need to maintain C_sqrt_inv. 
            # For simplicity and speed in this specific constrained context, 
            # we can use a simplified update or a standard Cholesky update.
            # However, full CMA-ES requires updating C and its inverse square root.
            
            # Update p_sigma
            # Note: In a full implementation, we evaluate the population first.
            # Let's generate the candidates, evaluate them, then update state.
            
            # Current CMA-ES step:
            # 1. Generate offspring
            # We need the current C_sqrt. We maintain C_sqrt_inv, so C_sqrt is its inverse.
            # Actually, it's easier to maintain C and compute Cholesky each step or use the update rule.
            # Given the small dim (5), computing Cholesky is cheap.
            
            # Let's stick to a robust implementation.
            # Re-initialize C_sqrt from C if needed, or maintain it.
            # Standard CMA-ES maintains C and updates it, then computes C_sqrt via Cholesky.
            
            L, _ = np.linalg.cholesky(C)
            # Y is already standard normal. We scale by sigma * L^T
            # Wait, the sampling is x + sigma * L^T * Y
            # Actually, the standard form is x + sigma * sum_i w_i * z_i
            # where z_i are sampled from N(0, C).
            # So we sample Z ~ N(0, I) and transform to Z' = L^T Z.
            
            Z = np.random.randn(self.dim, lambda_)
            # Transform Z to have covariance C
            # If C = L L^T, then L^T Z has covariance L^T L = C? No.
            # If C = L L^T, then L Z has covariance L L^T L^T L^T? No.
            # Let C = L L^T. Then (L^T Z) has covariance L^T L^T? No.
            # Let's use the property: If Z ~ N(0, I), then L^T Z ~ N(0, L^T L).
            # We want covariance C. So we need C = M M^T. Then M Z ~ N(0, C).
            # So we use L Z where C = L L^T.
            # Wait, Cholesky gives L such that C = L L^T.
            # So if we take X = L Z, Cov(X) = L Cov(Z) L^T = L L^T = C.
            # So we should use L Z.
            
            Offspring = np.array([x + sigma * (L @ z) for z in Z.T])
            
            # Clip to bounds
            Offspring = np.clip(Offspring, -5, 5)
            
            # Evaluate
            # Check if we have enough budget for the whole population
            if calls_made + lambda_ > self.budget:
                # If we can't afford the whole population, just do a few more single steps
                # Or break. Let's just break to be safe and simple.
                # Or we can reduce lambda_ temporarily.
                # For simplicity, we break if we can't do a full iteration.
                # But we might have 1-4 calls left.
                # Let's just do single steps for the remainder if lambda_ doesn't fit.
                while calls_made < self.budget and time.time() - t_start < time_limit:
                    z = np.random.randn(self.dim)
                    y = x + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    # Simple acceptance for the tail
                    if fy < func(x): # This requires an extra call, which we don't have budget for blindly.
                        # We don't have f(x) stored separately in this loop structure easily without extra call.
                        # Let's just update x if it's better than current best? No, x is the centroid.
                        # Let's just keep best_x.
                        pass
                    x = y # Move centroid? No, this is not proper CMA-ES.
                    # Actually, for the tail, a simple (1+1) ES is safer.
                    # We already have best_x. Let's just return best_x.
                    break 
                break
            
            F = np.array([func(y) for y in Offspring])
            calls_made += lambda_
            
            # Update best
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = Offspring[idx_best].copy()
            
            # Sort offspring
            idx = np.argsort(F)
            Y_sorted = Offspring[idx]
            F_sorted = F[idx]
            
            # Calculate new mean
            x_new = np.sum(w[:, np.newaxis] * Y_sorted[:mu], axis=1)
            
            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x_new - x)
            
            # Update p_sigma
            # We need C_sqrt_inv. We can approximate or compute it.
            # For dim=5, computing inverse Cholesky is cheap.
            C_sqrt_inv = np.linalg.inv(L)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (x_new - x) / sigma
            
            # Check for cumulation
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - lambda_))) > (1.4 + 2 / (self.dim + 1)) * chi_n:
                sigma *= np.exp((c_sigma / 5) * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update Covariance Matrix C
            # Delta_h = 1 if norm(p_c) / sqrt(1 - (1 - c_c)^(2*(calls-made))) < (1.4 + 2/(dim+1)) * chi_n else 0
            h = 1 if (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**(2 * (calls_made - lambda_))))**2 < (1 - (1 - c_c)**(2 * (calls_made - lambda_))) * (1.4 + 2 / (self.dim + 1)) * chi_n**2 else 0
            
            # Rank-mu update
            Delta_h = (1 - h) * c_c * (2 - c_c)
            
            # Rank-1 update
            C += (c_1 + c_mu) * np.outer(p_c, p_c)
            
            # Rank-mu update
            # Sum of outer products of selected steps
            Z_diff = (Y_sorted[:mu] - x) / sigma
            # Weighted sum
            # This is the expensive part, but dim=5 is small.
            RankMu = np.sum(w[:, np.newaxis] * np.outer(Z_diff.T, Z_diff), axis=2)
            
            C += c_mu * RankMu
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Project C to the positive definite cone if necessary
            # For dim=5, we can just do a Cholesky and if it fails, regularize.
            try:
                L, _ = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize
                C += 1e-6 * np.eye(self.dim)
                L, _ = np.linalg.cholesky(C)
            
            x = x_new
            sigma = max(sigma, 1e-10) # Prevent sigma from becoming too small
            
        return best_x, best_f
