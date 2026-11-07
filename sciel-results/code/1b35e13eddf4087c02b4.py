"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # For dim=5, lambda is usually 4 + floor(3 * log(5)) = 4 + 4 = 8
        # We use a slightly larger population for better exploration in early stages
        lambda_pop = 10
        
        # Initial mean: center of the box
        x = np.zeros(dim)
        # Initial sigma: large enough to explore the box [-5, 5]^5
        sigma = 3.0
        
        # CMA-ES adaptation parameters
        mu = lambda_pop // 2  # Number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, sqrt(mu_eff / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 + 2 / (dim + 1.3)) / ((dim + 1.3 + 2) * (dim + 1 + 2 / (mu_eff)))
        c_mu = (2 + weights[0] - c_1) / (dim + 5 + 2 / c_1)
        
        # Precompute factors
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        C = np.eye(dim)
        
        # Track best
        best_x = x.copy()
        best_f = func(best_x)
        calls_made = 1
        
        # Generate initial population
        # To save time, we can evaluate the initial population immediately
        # But we need to handle the budget carefully.
        
        # We will loop until budget is exhausted or time is up.
        # Each iteration uses `lambda_pop` calls.
        
        while calls_made < budget:
            if time.time() > deadline - 5.0:  # Leave 5s for finishing
                break
            
            # Generate offspring
            # Need to compute D = sqrt(diag(C))
            diag_C = np.diag(C)
            D = np.sqrt(diag_C)
            B, D_inv = np.linalg.eigh(C)
            B = B @ np.diag(1.0 / D_inv)
            
            # Sample z_i
            Z = rng.standard_normal((lambda_pop, dim))
            Y = x + sigma * (Z @ B.T)
            
            # Clip to box
            Y = np.clip(Y, -5, 5)
            
            # Evaluate
            F = np.array([func(y) for y in Y])
            calls_made += lambda_pop
            
            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = Y[idx_min].copy()
                
            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]
            
            # Compute weighted mean
            y_w = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)
            
            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (D_inv @ (y_w - x) / sigma))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2)) < (1.4 + 2 / (dim + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_w - x) / sigma
            
            # Compute rank-one and rank-mu updates
            # First, update C
            C = (1 + c_1 * (2 - c_1) + c_mu * (2 - c_mu) - c_1 - c_mu) * C
            
            # Rank-one update
            pc_outer = np.outer(pc, pc)
            C += c_1 * (pc_outer - C) # This is slightly simplified, standard is (1+c1*(2-c1))*C + c1*(pc@pc - C)
            # Let's use the standard formula properly:
            C = (1 + c_1 * (2 - c_1)) * C + c_1 * (np.outer(pc, pc) - C)
            
            # Rank-mu update
            Y_mu = Y_sorted[:mu]
            Y_mu_centered = Y_mu - y_w
            C += c_mu * np.sum(weights[:, None] * (Y_mu_centered[:, None, :] * Y_mu_centered[:, None, :].transpose(0, 2, 1)).reshape(dim, dim, -1), axis=2)
            
            # Update x and sigma
            x = y_w
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Ensure sigma is within reasonable bounds
            sigma = np.clip(sigma, 1e-10, 100.0)
            
            # Ensure C is positive definite (symmetrize and check)
            C = (C + C.T) / 2.0
            # If C becomes ill-conditioned, reset it
            if np.any(np.diag(C) < 1e-12):
                C = np.eye(dim)
                sigma = 1.0

        return best_x, best_f

# EVOLVE-BLOCK-END
