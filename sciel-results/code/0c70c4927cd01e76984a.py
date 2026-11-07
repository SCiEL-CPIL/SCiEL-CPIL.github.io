"""Rewritten optimizer using CMA-ES with covariance adaptation for BBOB f18."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES variant for BBOB-style f18 (Schaffers F7 ill-conditioned)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 280  # 280s out of 300s limit, leaving 20s margin

        # CMA-ES parameters
        # Population size
        lam = max(4, int(4 + 3 * np.log(d)))
        
        # Initial sigma
        # For f18 (ill-conditioned), start moderately small but not too small
        sigma = 1.0
        
        # Initial center
        # Random start in [-5, 5]
        mean = rng.uniform(-5, 5, d)
        
        # CMA-ES state
        C = np.eye(d)
        C_inv = np.eye(d)
        C_sqrt = np.eye(d)
        
        # Learning rates
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        
        # Step size control
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        
        # State variables
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Evaluation counter
        calls_made = 0
        best_x = mean.copy()
        best_f = float('inf')
        
        # Evaluate initial mean
        best_f = func(best_x)
        calls_made += 1
        
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            # We need to ensure we don't exceed budget
            # If remaining calls < lam, we might do a smaller population or stop
            remaining = budget - calls_made
            if remaining < lam:
                # Do a smaller population or single step
                # For simplicity, let's just do single steps if we have very few calls left
                # But CMA-ES is population based. Let's just break if we can't do a full generation
                # Or we can do a (1+1)-ES style step for the last few calls
                if remaining <= 0:
                    break
                # Do a single step (1+1) style
                y = np.clip(mean + sigma * C_sqrt @ rng.standard_normal(d), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    mean = y.copy()
                    # Update sigma simply
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                continue
            
            # Sample lam points
            points = np.zeros((lam, d))
            for i in range(lam):
                points[i] = np.clip(mean + sigma * C_sqrt @ rng.standard_normal(d), -5, 5)
            
            # Evaluate
            values = np.zeros(lam)
            for i in range(lam):
                if time.time() > deadline:
                    # Fill rest with inf and break
                    values[i:] = np.inf
                    break
                values[i] = func(points[i])
                calls_made += 1
            
            # Sort by fitness
            order = np.argsort(values)
            y_sorted = points[order]
            f_sorted = values[order]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()
                
            # Update mean
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
            
            # Update C
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_sqrt.T @ (mean - old_mean))
            
            # h_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (budget - calls_made + 1)) / chi_n if norm_p_sigma > 0 else 0
            # Note: The exponent should be the number of generations, but we don't track that exactly here.
            # Let's use a simpler approximation for h_sigma or just use a threshold.
            # Actually, the standard formula uses the number of iterations since last update.
            # Let's just use a heuristic: if norm_p_sigma is not too large compared to chi_n
            h_sigma = 1.0 if norm_p_sigma < (1.4 + 2 / (d + 1)) * chi_n else 0.0
            
            # p_c update
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (mean - old_mean)
            
            # C update
            delta_h = (2 - h_sigma) * c_c * (1 - c_c)
            C = (1 - c_1 - c_mu + delta_h) * C
            C += c_1 * (np.outer(p_c, p_c) - C)
            C += c_mu / sigma ** 2 * np.sum(weights[:, None] * np.outer(y_sorted[:mu] - old_mean, y_sorted[:mu] - old_mean), axis=0)
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Add small regularization to prevent singularity
            C += 1e-8 * np.eye(d)
            
            # Update C_sqrt and C_inv
            try:
                # Eigen decomposition for stable sqrt
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip eigenvalues to be positive
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            except np.linalg.LinAlgError:
                # Fallback
                C = np.eye(d)
                C_sqrt = np.eye(d)
                C_inv = np.eye(d)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # Check convergence
            if sigma < 1e-12:
                break
            if np.all(np.abs(C) < 1e-12):
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
