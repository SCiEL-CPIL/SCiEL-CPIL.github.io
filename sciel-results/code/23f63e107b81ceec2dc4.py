"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive covariance matrix for f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # lambda (population size)
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2  # number of parents
        
        # Weights
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        w = w**0.8  # Sharpen weights slightly for faster convergence in low dim
        w /= w.sum()
        
        mueff = 1 / np.sum(w**2)
        
        # Learning rates
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Initial sigma
        # f11 has a wide valley. Starting with a large sigma helps find the valley.
        # The box is [-5, 5]. The minimum is in [-4, 4].
        # A sigma of 2.0 is reasonable.
        sigma = 2.0
        
        # Initial center
        # Random start within the box, but maybe closer to center is safer?
        # The task says hidden shift/rotation. Uniform start is unbiased.
        rng = np.random.default_rng(42)
        x_mean = rng.uniform(-3, 3, n)
        
        # Covariance matrix
        C = np.eye(n)
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Precompute
        hsigma = (1 + 0.3 * (mueff / n + 1)) * np.sqrt((n + 1) * cc * (2 - cc))
        
        # Call 1
        x0 = np.clip(x_mean, -5, 5)
        f_best = func(x0)
        best_x = x0.copy()
        calls_made = 1
        
        # We need to ensure we don't exceed budget.
        # CMA-ES usually evaluates lambda points per generation.
        # Total budget is 500.
        # We will run generations until budget is exhausted or time runs out.
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            # Ensure we don't exceed budget in this generation
            if calls_made + lam > budget:
                lam_eff = budget - calls_made
            else:
                lam_eff = lam
                
            if lam_eff <= 0:
                break

            # Adjust mu if lam_eff is small? 
            # For simplicity, if lam_eff < mu, we just evaluate what we can and use top min(mu, lam_eff)
            # But standard CMA-ES assumes fixed lam. 
            # Given budget 500 and lam ~ 15-20, this is fine.
            
            # Sample from N(x_mean, sigma^2 C)
            # Cholesky factorization of C
            L = np.linalg.cholesky(C)
            
            ys = []
            fs = []
            
            for i in range(lam_eff):
                z = rng.standard_normal(n)
                y = x_mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                ys.append(y)
                fs.append(fy)
                calls_made += 1
                
                if time.time() > deadline:
                    break
            
            if len(ys) == 0:
                break

            ys = np.array(ys)
            fs = np.array(fs)
            
            # Update best
            idx_min = np.argmin(fs)
            if fs[idx_min] < f_best:
                f_best = fs[idx_min]
                best_x = ys[idx_min].copy()
            
            # Sort by fitness
            idx_sorted = np.argsort(fs)
            ys = ys[idx_sorted]
            fs = fs[idx_sorted]
            
            # Effective mu for this generation
            mu_eff = min(mu, len(ys))
            if mu_eff < 1:
                break
            
            # Recalculate weights for this effective mu? 
            # Standard CMA-ES uses fixed mu. If we have fewer points, we just use the available ones.
            # To keep it simple and robust, if lam_eff < mu, we just proceed with available points 
            # but the math for p_sigma/p_c assumes specific mu. 
            # Given budget 500 and lam ~ 15, we likely won't hit the tail end with lam_eff < mu often 
            # unless we are very close to budget. 
            # Let's just use the first mu_eff points.
            
            # Weights for the effective mu_eff
            if mu_eff < mu:
                 w_eff = np.log(np.arange(1, mu_eff + 1))
                 w_eff /= w_eff.sum()
                 w_eff = w_eff**0.8
                 w_eff /= w_eff.sum()
                 mueff_eff = 1 / np.sum(w_eff**2)
            else:
                 w_eff = w
                 mueff_eff = mueff
                 
            # Update mean
            x_mean_new = np.sum(w_eff[:, np.newaxis] * ys[:mu_eff], axis=0)
            
            # Recalculate C inverse for p_sigma update? 
            # p_sigma update uses C^-0.5. 
            # We can compute C^-0.5 via eigendecomposition or just use the previous C for the step.
            # Standard algorithm uses C^-0.5 of the *current* C (before update).
            
            # Eigen decomposition for C^-0.5
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals_inv_sqrt = 1 / np.sqrt(np.maximum(eigvals, 1e-12))
            C_inv_sqrt = (eigvecs * eigvals_inv_sqrt) @ eigvecs.T
            
            # Update p_sigma
            d_mean = (x_mean_new - x_mean) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff_eff) * C_inv_sqrt @ d_mean
            
            # Update sigma (CSA)
            sigma *= np.exp(cs / hsigma * (np.linalg.norm(p_sigma) / np.sqrt(n) - 1))
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update p_c
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff_eff) * (x_mean_new - x_mean) / sigma
            
            # Update C
            # Outer product of weighted differences
            diff = (x_mean_new - x_mean) / sigma
            outer = np.outer(diff, diff)
            
            # Sum of weighted outer products of steps
            steps = (ys[:mu_eff] - x_mean) / sigma
            # W is (mu_eff, 1), steps is (mu_eff, n)
            # We need sum w_i * (y_i - x_mean) @ (y_i - x_mean)^T
            # steps.T @ (w_eff[:, None] * steps) is not quite right for broadcasting if w is 1D.
            # Let's do: steps.T @ np.diag(w_eff) @ steps
            # Or: (steps * w_eff[:, np.newaxis]).T @ steps
            
            sum_steps = (steps * w_eff[:, np.newaxis]).T @ steps
            
            C = (1 + c1 * (np.outer(diff, diff) - C) + cmu * (sum_steps - C)) * (1 - c1 - cmu) # Wait, the formula is C = (1-c1-cmu)C + c1(...)+cmu(...)
            # Correct formula: C = (1 - c1 - cmu) * C + c1 * (diff @ diff.T) + cmu * sum_steps
            # Note: c1 + cmu is usually small, so (1-c1-cmu) is close to 1.
            # Let's re-apply correctly.
            C = (1 - c1 - cmu) * C + c1 * (np.outer(diff, diff)) + cmu * sum_steps
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = x_mean_new
            
            # Check for convergence
            if np.linalg.norm(x_mean - best_x) < 1e-10:
                break

        return best_x, f_best
# EVOLVE-BLOCK-END
