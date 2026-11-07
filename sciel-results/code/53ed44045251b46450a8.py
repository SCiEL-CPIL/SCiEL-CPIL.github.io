"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time-guarded execution."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 300s limit + 20s safety margin

        rng = np.random.default_rng(42)
        dim = self.dim

        # CMA-ES parameters
        # Population size: 4 + 3*floor(0.5 + log(1 + 4*dim))
        # For dim=5: 4 + 3*floor(0.5 + log(21)) ≈ 4 + 3*4 = 16? 
        # Standard formula: lambda = 4 + 3 * floor(0.5 + log(1 + 4 * dim))
        # log(21) ≈ 3.04, 0.5+3.04=3.54, floor=3, so lambda = 4+9=13? 
        # Let's use a smaller population for budget efficiency. 
        # Budget is 500. 
        # If lambda=4, iterations=125. If lambda=8, iterations=62.
        # Let's use lambda=4 for more iterations (more adaptation steps).
        # Or maybe lambda=6? 
        # Let's stick to standard small lambda for high dimensionality? No, dim=5 is low.
        # Standard CMA-ES often uses lambda ~ 4 + 3*log(1+4*dim).
        # For dim=5, log(21)=3.04, lambda=13. 500/13 = 38 iterations.
        # That might be too few for adaptation.
        # Let's use a smaller population, e.g., lambda=4 or 6, to allow more generations.
        # Actually, for dim=5, lambda=4 is quite small but allows 125 generations.
        # Let's try lambda=4.
        
        lambda_ = 4
        mu = lambda_ // 2  # Weighting for top half
        # Weights: rank-based
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial values
        x_mean = np.zeros(dim)
        # Initial sigma: heuristic based on domain size
        # Domain is [-5, 5], so size is 10. 
        # Standard initial sigma is often 0.5 * range or similar.
        # Let's start with sigma = 2.0 (as in seed) or maybe 1.0?
        # BBOB functions often have minima in [-4,4]. 
        # Starting at 0 with sigma=2 covers the space well.
        sigma = 2.0
        
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = np.inf
        
        # Precompute sqrt(C) via Cholesky or Eigen decomposition
        # We will update C and recompute sqrt each generation.
        
        # Initial evaluation
        f_vals = np.full(lambda_, np.inf)
        # Evaluate initial population? 
        # Standard CMA-ES evaluates lambda new points each generation.
        # Total budget 500.
        # Let's do 1 initial eval + loop.
        
        # To save budget, we can initialize x_mean randomly? 
        # No, x_mean=0 is a good prior for BBOB (shifted/rotated but usually centered).
        # But wait, the task says "hidden seeded shift". The minimizer is in [-4,4]^5.
        # Starting at 0 is safe.
        
        # We need to be careful with the budget.
        # Each generation costs lambda_ evaluations.
        # Max generations = (budget - 1) // lambda_ ?
        # Let's just loop until budget or time.
        
        eval_count = 0
        max_evals = self.budget
        
        # Helper to evaluate and update best
        def evaluate(x):
            nonlocal best_x, best_f, eval_count
            if eval_count >= max_evals:
                return np.inf
            eval_count += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        # Initial point
        evaluate(x_mean)
        
        while eval_count < max_evals and time.time() < deadline:
            # Sample population
            # Need sqrt(C)
            try:
                # Use Cholesky for speed, fallback to eigh
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                w, V = np.linalg.eigh(C)
                L = V @ np.diag(np.sqrt(np.maximum(w, 0)))
            
            z = rng.standard_normal((lambda_, dim))
            pop = x_mean + sigma * (z @ L.T)
            
            # Clip to box? 
            # The function clips internally, but for CMA-ES, clipping can distort the distribution.
            # However, the domain is bounded. 
            # It is better to let the function handle it, or clip x to [-5,5] to avoid out-of-bounds if func doesn't? 
            # "func(x) clips a finite length-5 vector to the box". So func handles it.
            # We should NOT clip x before passing to func, as it might break the symmetry of the sampling if the mean is near the boundary? 
            # Actually, if x_mean is far, it doesn't matter. 
            # But CMA-ES assumes unbounded search. 
            # If we clip, the effective distribution changes. 
            # Given "func(x) clips", we can pass unclipped x.
            
            f_new = np.empty(lambda_)
            for i in range(lambda_):
                if eval_count >= max_evals:
                    break
                f_new[i] = evaluate(pop[i])
            
            if eval_count >= max_evals:
                break
                
            # Sort by fitness
            idx = np.argsort(f_new)
            x_sorted = pop[idx]
            f_sorted = f_new[idx]
            
            # Update weights if necessary? No, fixed weights.
            w = weights
            
            # New mean
            x_new_mean = np.sum(w[:, None] * x_sorted[:mu], axis=0)
            
            # Update step-size control variables
            # ps = (1 - c_sigma) * ps + c_sigma * sqrt(mu_eff) * C^{-1/2} * (x_new_mean - x_mean) / sigma
            # We need C^{-1/2}. 
            # Instead of computing inverse sqrt, we can use the fact that we have L.
            # C^{-1/2} = L^{-T}
            # Let's solve L^T y = (x_new_mean - x_mean) / sigma
            diff = (x_new_mean - x_mean) / sigma
            try:
                # Solve L^T y = diff
                y = np.linalg.solve(L.T, diff)
            except np.linalg.LinAlgError:
                y = diff # Fallback
            
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) * y
            
            # Update evolution path pc
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * (x_new_mean - x_mean) / sigma
            
            # Update covariance matrix
            # C = (1 - c_1 - c_mu * h_sigma) * C + c_1 * pc * pc.T + c_mu * sum(w_i * (y_i - x_mean)(y_i - x_mean).T)
            
            # h_sigma: Heaviside function for step-size adaptation
            norm_ps = np.linalg.norm(ps)
            # Expected norm of ps
            E_norm_ps = np.sqrt(mu_eff) * (1 + 1 / (2 * mu_eff) + 1 / (8 * mu_eff**2))
            h_sigma = (norm_ps / np.sqrt(1 - (1 - c_sigma)**(2 * (eval_count - 1)))) if (1 - c_sigma)**(2 * (eval_count - 1)) > 1e-10 else 1.0
            # Simplified h_sigma check:
            # h_sigma = 1 if norm_ps / sqrt(1 - (1-c_sigma)^(2t)) < c1 + 1/c1 * sqrt((1+c1)/(1-c1)) * E_norm_ps
            # This is complex. Let's use a simpler threshold or standard implementation.
            # Standard: h_sigma = 1 if (norm_ps / sqrt(1 - (1-c_sigma)^(2t))) < (1 + 1/sqrt(mu_eff)) * E_norm_ps
            # Let's approximate t = eval_count / lambda_
            t = eval_count / lambda_
            denom = np.sqrt(1 - (1 - c_sigma)**(2 * t)) if t > 0 else 1.0
            if denom < 1e-10:
                h_sigma = 0
            else:
                threshold = (1 + 1 / np.sqrt(mu_eff)) * E_norm_ps
                h_sigma = 1.0 if (norm_ps / denom) < threshold else 0.0
            
            # Compute covariance update
            # We need the weighted sum of outer products of (x_i - x_mean)
            # Let's use the sorted population
            diff_pop = (x_sorted[:mu] - x_mean) / sigma
            # Weighted sum
            c_sum = np.sum(w[:, None] * (diff_pop * diff_pop.T), axis=0)
            
            # Update C
            C = (1 - c_1 - c_mu * h_sigma) * C + c_1 * np.outer(pc, pc) + c_mu * c_sum
            
            # Ensure C is symmetric and positive semi-definite
            C = (C + C.T) / 2.0
            # Add small regularization if needed?
            # C += 1e-10 * np.eye(dim)
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||ps|| / E[||N(0, I)||] - 1))
            d_sigma = 1 + 2 * max(0, (mu_eff - 2) / (dim + mu_eff - 2)) + c_sigma
            sigma *= np.exp((c_sigma / d_sigma) * (norm_ps / E_norm_ps - 1))
            
            # Update mean
            x_mean = x_new_mean
            
            # Check for convergence or stagnation?
            # If sigma is very small, stop?
            if sigma < 1e-8:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
