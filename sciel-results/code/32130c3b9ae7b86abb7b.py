"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 320.0

        # CMA-ES parameters
        # Population size
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        
        # Strategy parameters
        c_sigma = (lambda_ + 2) / (dim + lambda_ + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_ - 1) / (dim + 1)) - 1) + c_sigma
        chi_lambda = np.sqrt(lambda_) * (1 - 1 / (4 * dim) + 1 / (20 * lambda_**2))
        
        # Mean update parameters
        c_c = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        c_1 = 2 / ((dim + 1.3)**2 + lambda_)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / ((dim + 2) * (mu_eff + 1.5)) - c_1))
        
        # Weighted average
        mu = int(lambda_ / 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        # Recalculate c_mu with mu_eff
        c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / ((dim + 2) * (mu_eff + 1.5)) - c_1))
        
        # Restart logic
        best_x = None
        best_fx = np.inf
        
        total_evals = 0
        restart_count = 0
        max_restarts = 10
        
        while total_evals < budget and restart_count < max_restarts and time.time() < deadline:
            # Initialize CMA-ES
            # Start from a random point or best found so far
            if best_x is None:
                mean = rng.uniform(-5, 5, dim)
            else:
                # Perturb best slightly for restart
                mean = best_x + 0.5 * rng.standard_normal(dim)
                mean = np.clip(mean, -5, 5)
            
            sigma = 2.0
            C = np.eye(dim)
            C_sqrt_inv = np.eye(dim)
            C_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Initial population
            evals_in_restart = 0
            stuck = 0
            best_fx_restart = np.inf
            best_x_restart = None
            
            # Evaluate initial mean
            fx_mean = func(mean)
            total_evals += 1
            if fx_mean < best_fx:
                best_fx = fx_mean
                best_x = mean.copy()
            if fx_mean < best_fx_restart:
                best_fx_restart = fx_mean
                best_x_restart = mean.copy()
            
            # Main loop for this restart
            while total_evals < budget and evals_in_restart < 1000 and time.time() < deadline:
                # Check termination based on stagnation
                if stuck > 50:
                    break
                
                # Sample population
                # Update C_sqrt_inv (slowly)
                # For simplicity in 5D, we can use a cheaper update or just fixed step for C update
                # But CMA-ES needs proper update.
                
                # Sample
                z = rng.standard_normal((lambda_, dim))
                # Mahalanobis transform: x = mean + sigma * C_sqrt @ z
                # We need C_sqrt. Initially I.
                # To keep it simple and fast, we might skip full C update if it's too slow, 
                # but for 5D it's fine.
                
                # Update C_sqrt_inv
                # h_sigma = (||p_sigma|| / sqrt(1 - (1-c_sigma)^{2*(gen+1)})) / chi_lambda < 1.4 + 2/(dim+1)
                # This is complex. Let's use a simplified CMA-ES or standard one.
                
                # Standard CMA-ES update
                # 1. Sample
                z = rng.standard_normal((lambda_, dim))
                # We need C_sqrt. Let's maintain it.
                # Initially C=I, C_sqrt=I.
                # x = mean + sigma * (C_sqrt @ z.T).T
                x = mean + sigma * (C_sqrt @ z.T).T
                
                # Clip to box
                x = np.clip(x, -5, 5)
                
                # Evaluate
                fvals = np.array([func(xi) for xi in x])
                total_evals += lambda_
                evals_in_restart += lambda_
                
                # Sort
                idx = np.argsort(fvals)
                x = x[idx]
                fvals = fvals[idx]
                
                # Update best
                if fvals[0] < best_fx:
                    best_fx = fvals[0]
                    best_x = x[0].copy()
                if fvals[0] < best_fx_restart:
                    best_fx_restart = fvals[0]
                    best_x_restart = x[0].copy()
                    stuck = 0
                else:
                    stuck += 1
                
                # Weighted mean
                x_w = np.sum(weights[:, np.newaxis] * x[:mu], axis=0)
                
                # Update covariance matrix
                # p_sigma update
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ (x_w - mean) / sigma)
                
                # h_sigma
                # Simplified: assume h_sigma = 1 if norm is small
                # Proper:
                # denom = np.sqrt(1 - (1 - c_sigma)**(2 * (evals_in_restart + 1)))
                # if denom > 0:
                #     h_sigma = np.linalg.norm(p_sigma) / (chi_lambda * denom) < 1.4 + 2 / (dim + 1)
                # else:
                #     h_sigma = 1
                # For speed and simplicity in this short budget, let's use a simplified check or skip h_sigma complexity if it's too slow.
                # Given 5D, let's implement properly but efficiently.
                
                # p_c update
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_sqrt_inv @ (x_w - mean) / sigma)
                
                # C update
                # Delta_h = h_sigma - (1 - c_c)
                # C = (1 + c_1 + c_mu - c_1 * (1 - h_sigma)) * C + c_1 * (p_c @ p_c.T - C) + c_mu * sum(weights * (x_i - mean)/sigma @ (x_i - mean)/sigma.T)
                
                # Compute C_sqrt_inv and C_sqrt
                # Eigen decomposition of C
                # C is symmetric positive definite
                # Use Cholesky or Eigen
                # Eigen is more stable for updates
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip eigenvalues to be positive
                eigvals = np.clip(eigvals, 1e-12, None)
                
                # Update C
                # Term 1
                C_new = (1 + c_1 + c_mu - c_1 * h_sigma) * C
                # Term 2
                C_new += c_1 * (p_c @ p_c.T - C)
                # Term 3
                # Sum of weighted outer products
                # (x[:mu] - mean) / sigma
                delta_x = (x[:mu] - mean) / sigma
                # C_sqrt_inv @ delta_x
                # We need C_sqrt_inv.
                # C = V @ diag(w) @ V.T
                # C_sqrt = V @ diag(sqrt(w)) @ V.T
                # C_sqrt_inv = V @ diag(1/sqrt(w)) @ V.T
                
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                # delta_x_trans = C_sqrt_inv @ delta_x
                delta_x_trans = (C_sqrt_inv @ delta_x.T).T
                # Sum of weights * outer(delta_x_trans, delta_x_trans)
                # This is sum_i w_i * (d_i @ d_i.T)
                # Can be computed as (delta_x_trans.T * weights) @ delta_x_trans
                C_new += c_mu * ((delta_x_trans.T * weights) @ delta_x_trans)
                
                C = C_new
                
                # Update mean
                mean = x_w
                mean = np.clip(mean, -5, 5)
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_lambda - 1))
                # Clamp sigma
                if sigma < 1e-10:
                    break
                if sigma > 100:
                    sigma = 100
                
                # Update C_sqrt and C_sqrt_inv for next iteration
                # We already have them from the update step, but they correspond to old C?
                # No, C was updated to C_new. We need new C_sqrt.
                # Re-compute or update incrementally.
                # For 5D, re-computing is fine.
                # But we did it inside the C update block.
                # So C_sqrt and C_sqrt_inv are ready for next sampling.
                
                # Check time
                if time.time() > deadline:
                    break
            
            restart_count += 1
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
