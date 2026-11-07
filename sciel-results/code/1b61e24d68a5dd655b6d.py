"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with recombination (cross) operator for anisotropic landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        # Parameters
        n = self.dim
        lam = 9
        mu = 4
        c = 1 / (n + 3)
        c_sigma = (n + 1.3) / (n + 3 + 2)
        c_cov = (2 + 2 * min(mu / (n + 1), 1) - c - c_sigma) / (n + 4)
        d_cov = 5 + n / 2 - c_cov
        d_sigma = 1 + 2 / max(n - 1, 1) + min(mu / (n + 1), 1)
        
        # Weights and sigma weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        # Initial state
        rng = np.random.default_rng(0)
        # Start near center, as minimizer is in [-4, 4]^5
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        cs = np.sqrt((c_sigma + 1) * (n + 2))
        chisigma = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        budget_used = 0
        best_x = x_mean.copy()
        best_f = np.inf
        
        # Initial evaluation
        f = func(x_mean)
        budget_used += 1
        best_f = f
        best_x = x_mean.copy()
        
        while budget_used < self.budget:
            if time.time() > deadline:
                break
                
            # Generate population
            B = rng.standard_normal((lam, n))
            # Cholesky of C
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes non-positive definite
                C = np.eye(n)
                C_chol = np.eye(n)
                
            Y = (x_mean + sigma * (B @ C_chol.T))
            Y = np.clip(Y, -5, 5)
            
            # Evaluate
            Fs = np.array([func(y) for y in Y])
            budget_used += lam
            
            # Update best
            min_idx = np.argmin(Fs)
            if Fs[min_idx] < best_f:
                best_f = Fs[min_idx]
                best_x = Y[min_idx].copy()
                
            # Sort by fitness
            order = np.argsort(Fs)
            Y_sorted = Y[order]
            
            # Recombination (Cross operator)
            y_w = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)
            
            # Update sigma (CSA)
            ps = (1 - c_sigma) * ps + cs * np.sqrt(c_sigma * (2 - c_sigma) / mu_eff) * (C_chol.T @ (y_w / sigma))
            if np.linalg.norm(ps) / chisigma < 4.77 + 2 / n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chisigma - 1))
            
            # Update C (CMA)
            pc = (1 - c) * pc + c * np.sqrt(mu_eff) * (y_w - x_mean) / sigma
            v = c * (1 - c) * pc @ pc
            hsig = (1 - np.exp(-2 * (1 - c) * (1 + c_sigma) / d_cov)) * np.linalg.norm(pc) / chisigma
            C = (1 + c_cov * (mu_eff - 2 + hsig) - c) * C + c_cov * (np.outer(y_w, y_w) / sigma**2 + (hsig - 1) * v)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = y_w
            x_mean = np.clip(x_mean, -5, 5)
            
            # Ensure sigma doesn't get too small or too large
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # If we are very close to the boundary or sigma is tiny, we might be stuck
            # But CMA-ES handles this better than (1+1)-ES.
            
            # Check if we have used all budget
            if budget_used >= self.budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
