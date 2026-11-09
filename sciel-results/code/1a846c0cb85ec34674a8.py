"""Optimized program for BBOB f14 using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 290s out of 300s limit
        deadline = start_time + 290
        
        n = self.dim
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(n))
        if lambda_ < 4:
            lambda_ = 4
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / (weights ** 2).sum()
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Initialization
        x = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        # C_inv_sqrt is inverse of Cholesky factor
        C_inv_sqrt = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        best_x = x.copy()
        best_f = func(x)
        calls = 1
        k = 0  # Iteration counter
        
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # Sample lambda_ points
            z = np.random.randn(lambda_, n)
            # y = x + sigma * C_inv_sqrt @ z.T
            # Note: C_inv_sqrt @ z.T gives (n, lambda_)
            # We need (lambda_, n)
            y = x[None, :] + sigma * (C_inv_sqrt @ z.T).T
            
            # Evaluate and clip
            f_vals = np.zeros(lambda_)
            for i in range(lambda_):
                y[i] = np.clip(y[i], -5, 5)
                f_vals[i] = func(y[i])
            calls += lambda_
            
            # Sort by fitness
            idx = np.argsort(f_vals)
            y = y[idx]
            f_vals = f_vals[idx]
            
            # Update best
            if f_vals[0] < best_f:
                best_f = f_vals[0]
                best_x = y[0].copy()
            
            # Compute new mean
            x_new = np.sum(weights[:, None] * y[:mu], axis=0)
            
            # Update path memories
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new - x) / sigma
            
            # h_sigma calculation
            # Norm of ps
            ps_norm = np.linalg.norm(ps)
            # Denominator: sqrt(1 - (1-c_sigma)^(2*lambda_))
            # This is the expected norm of the path if it were a random walk
            # Actually, the standard formula is:
            # h_sigma = ||ps|| / sqrt(1 - (1-c_sigma)^(2*lambda_)) < (1 + 2/(n+1)) * sqrt(1 - (1-c_sigma)^2)
            # However, for stability, we can use a simplified check or the full one.
            # Let's use the full one.
            denom = np.sqrt(1 - (1 - c_sigma) ** (2 * lambda_))
            if denom == 0:
                h_sigma = 1
            else:
                h_sigma = int(ps_norm / denom < (1 + 2 / (n + 1)) * np.sqrt(1 - (1 - c_sigma) ** 2))
            
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            
            # Update Covariance Matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta_h * C)
            
            # Add rank-mu update
            for i in range(mu):
                zi = C_inv_sqrt @ (y[i] - x) / sigma
                C += c_mu * weights[i] * np.outer(zi, zi)
            
            # Update C_inv_sqrt
            try:
                L = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(L.T)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset
                C = np.eye(n)
                C_inv_sqrt = np.eye(n)
                sigma *= 0.5
                pc = np.zeros(n)
                ps = np.zeros(n)
            
            x = x_new
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (ps_norm / denom - 1))
            
            k += 1
            
        return best_x, best_f
# EVOLVE-BLOCK-END
