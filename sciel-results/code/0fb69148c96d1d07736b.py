"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        # For bent cigar, we want a relatively large initial sigma
        # and a fast-adapting covariance matrix.
        
        # Initial point: center of the box
        x = np.zeros(n)
        f_x = func(x)
        
        # Population size
        lambda_ = 4 + int(3 * np.log(n))
        mu = lambda_ // 2
        
        # Weights for the mu best points
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial sigma: for bent cigar, a larger sigma helps explore the valley
        sigma = 1.0
        
        # Covariance matrix and its square root (Cholesky)
        C = np.eye(n)
        C_inv = np.eye(n)
        # We'll maintain C via updates, and use its eigen-decomposition for sampling
        # For efficiency, we maintain the evolution path in C-space
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # History for tracking
        best_x = x.copy()
        best_f = f_x
        
        # Precompute some constants
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # We'll use a simplified CMA-ES that updates C via eigen-decomposition
        # This is slower per step but more stable. Given budget=500 and n=5,
        # lambda_ is about 4+3*ln(5) ≈ 4+4.8 ≈ 8-9. Let's use lambda_=9.
        # Actually, let's use a smaller population for speed: lambda_=6
        lambda_ = 6
        mu = 3
        weights = np.array([0.5, 0.3, 0.2])
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # For bent cigar, start with a larger sigma
        sigma = 2.0
        
        C = np.eye(n)
        # We'll maintain C and its Cholesky factor B such that C = B B^T
        B = np.eye(n)
        
        calls_made = 1  # we already made one call
        
        for gen in range(1, budget):
            if time.time() > deadline:
                break
            
            # Generate offspring
            y = []
            f_y = []
            for i in range(lambda_):
                # Sample from N(0, C) using B
                z = np.random.standard_normal(n)
                y_i = x + sigma * (B @ z)
                y_i = np.clip(y_i, -5, 5)
                f_y_i = func(y_i)
                calls_made += 1
                y.append(y_i)
                f_y.append(f_y_i)
                
                if f_y_i < best_f:
                    best_f = f_y_i
                    best_x = y_i.copy()
                
                # Check time budget inside the loop too
                if time.time() > deadline:
                    break
            
            # If we broke out early, pad with the best point
            while len(y) < lambda_:
                y.append(best_x.copy())
                f_y.append(best_f)
            
            # Sort offspring
            order = np.argsort(f_y)
            y = [y[i] for i in order]
            f_y = [f_y[i] for i in order]
            
            # Update evolution paths
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (np.mean([y[i] - x for i in range(mu)], axis=0) / sigma))
            
            # p_c update
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * chi_n
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ (np.mean([y[i] - x for i in range(mu)], axis=0) / sigma))
            
            # Update C
            # Rank-1 update
            y_mean = np.mean([y[i] for i in range(mu)], axis=0)
            diff = (y_mean - x) / sigma
            C = (1 + c_1 + c_mu - c_c * (2 - c_c) * (h_sigma - (1 - c_c))) * C + c_1 * (np.outer(p_c, p_c) - C)
            
            # Rank-mu update
            diff_matrix = np.array([y[i] - x for i in range(mu)]) / sigma
            C = C + c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.einsum('ij,ik->ijk', diff_matrix, diff_matrix), axis=0)
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update Cholesky factor
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: use eigen-decomposition
                B = eigvecs @ np.diag(np.sqrt(eigvals))
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            x = y_mean
            
            # Adaptive sigma based on progress (optional)
            # For bent cigar, we can increase sigma when progress is slow
            if gen > 10 and gen % 10 == 0:
                # Check if we've been making little progress
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
