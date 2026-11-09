"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f9 (Rotated Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # State
        best_x = np.zeros(n)
        best_f = func(best_x)
        calls_used = 1

        def check_deadline():
            return time.time() >= deadline

        def do_call(x):
            nonlocal calls_used
            fx = func(x)
            calls_used += 1
            return fx

        # CMA-ES Parameters
        # Pop size: lambda = 4 + floor(3 * log(n)) is standard, but for n=5, lambda=9 is small.
        # Let's use lambda=20 for better exploration initially, or stick to standard.
        # For n=5, lambda=9 is quite small. Let's use lambda=12 to be safe.
        pop_size = 12
        
        # Initial mean: Origin is a good start for BBOB f9 usually, or random?
        # BBOB f9 is rotated Rosenbrock. The global minimum is at the origin in the 
        # transformed space, but shifted/rotated. 
        # Starting at 0 is a reasonable heuristic for BBOB.
        mean = np.zeros(n)
        
        # Initial sigma: The box is [-5, 5]. The minimum is in [-4, 4].
        # A sigma of 2.0 is a good start to cover the space.
        sigma = 2.0
        
        # Covariance matrix C
        C = np.eye(n)
        
        # Step size control state
        # For simplicity and robustness, we use the standard CMA-ES update rules.
        # However, implementing full CMA-ES is complex. 
        # Alternative: Use a simpler adaptive strategy like (1+1)-ES with adaptive sigma 
        # but with a twist: use a population to estimate the covariance roughly?
        
        # Let's implement a simplified but effective version:
        # 1. Sample population from N(mean, sigma^2 * C)
        # 2. Evaluate
        # 3. Update mean using weighted sum of top performers (intermediate lambda weighting)
        # 4. Update C based on the history of successful steps
        # 5. Update sigma based on the ratio of step lengths
        
        # To keep it simple and robust within the budget, we might use a 
        # "CMA-like" update with a smaller population and careful restarts.
        
        # Actually, for n=5 and budget=500, a full CMA-ES is very fast.
        # Let's implement a standard CMA-ES.
        
        # Parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, sqrt(mu_eff / (n + 1) - 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Weights
        w = np.log(np.arange(1, pop_size + 1)) - np.log(pop_size)
        w = w / np.sum(w)
        mu = int(pop_size / 2)  # Use top half for updates
        w = w[:mu]
        mu_eff = 1.0 / np.sum(w**2)
        
        # Recalculate constants with mu_eff
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt(mu_eff / (n + 1) - 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Precompute eigenvalues? No, update C directly.
        
        # Helper to sample
        def sample():
            # Cholesky decomposition of C for sampling
            # Since C is updated, we need to be careful with numerical stability.
            # For n=5, this is cheap.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity
                C[:] = np.eye(n)
                L = np.eye(n)
            
            # Sample standard normal
            z = rng.standard_normal((pop_size, n))
            # Transform
            return mean + sigma * (z @ L.T)
        
        # Main Loop
        # We have 500 calls. 
        # Initialization: 1 call (done).
        # Each generation: pop_size calls.
        # 500 / 12 ~ 41 generations.
        
        gen = 0
        while calls_used + pop_size <= budget and not check_deadline():
            gen += 1
            
            # Sample
            x_pop = sample()
            # Clip to bounds
            x_pop = np.clip(x_pop, -5, 5)
            
            # Evaluate
            f_pop = np.array([do_call(x) for x in x_pop])
            
            # Update best
            idx_best_pop = np.argmin(f_pop)
            if f_pop[idx_best_pop] < best_f:
                best_f = f_pop[idx_best_pop]
                best_x = x_pop[idx_best_pop].copy()
            
            # Sort population by fitness
            idx_sorted = np.argsort(f_pop)
            x_sorted = x_pop[idx_sorted]
            f_sorted = f_pop[idx_sorted]
            
            # Update mean
            # Weighted mean of top mu
            mean_old = mean.copy()
            mean = np.sum(w[:, None] * x_sorted[:mu], axis=0)
            
            # Update p_sigma
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^(-1/2) * (mean - mean_old) / sigma
            # C^(-1/2) is needed. 
            # For simplicity and robustness in a short script, we might approximate 
            # or just use the step direction if C is close to identity.
            # But CMA-ES relies on this.
            # Let's compute C^(-1/2) via eigen decomposition or cholesky.
            # Since we have L from cholesky, C = L L^T. 
            # C^(-1/2) = (L L^T)^(-1/2).
            # This is tricky to compute directly without extra matrices.
            # Alternative: Use the "CMA-ES with diagonal covariance" (CMA-diag) 
            # which is much simpler and often sufficient for n=5.
            # CMA-diag only tracks variances, not correlations.
            # Given the rotation, correlations are important.
            # However, implementing full CMA-ES correctly is error-prone.
            
            # Let's switch strategy to a simpler but effective adaptive method:
            # **Adaptive (1+1)-ES with population-based sigma adaptation (AMS)**
            # This is robust, fast, and handles rotations reasonably well if we 
            # use a population to estimate the local landscape direction.
            
            # Actually, let's try a **Nelder-Mead with adaptive restarts** approach 
            # but starting from multiple random points? No, that's what the parent did poorly.
            
            # Let's stick to the CMA-ES idea but simplify the update.
            # We can update C using the "rank-one" and "rank-mu" updates.
            # y = (mean - mean_old) / sigma
            y = (mean - mean_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            
            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            
            # Update C
            # Rank-one update
            C = (1 + c_1 * (mu_eff - 1) - c_1) * C + c_1 * np.outer(y, y)
            
            # Rank-mu update
            # y_i = (x_i - mean_old) / sigma
            y_mu = (x_sorted[:mu] - mean_old) / sigma
            C += c_mu * np.sum(w[:, None, None] * np.einsum('i,ij,ik->jk', y_mu, y_mu, y_mu), axis=0)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization to ensure PD
            C += 1e-10 * np.eye(n)
            
            # Update sigma
            # h_sigma = 1 if ||p_sigma|| < (1.4 + 1/n) * sqrt((mu_eff - 1) + 1) * sqrt(c_sigma * (2 - c_sigma)) else 0
            # Actually, the condition is:
            # ||p_sigma|| / sqrt(1 - (1 - c_sigma)^{2 * gen}) < (1.4 + 1/n) * sqrt((mu_eff - 1) + 1) * sqrt(c_sigma * (2 - c_sigma))
            # For simplicity, we approximate the threshold.
            norm_p_sigma = np.linalg.norm(p_sigma)
            threshold = (1.4 + 1/n) * np.sqrt((mu_eff - 1) + 1) * np.sqrt(c_sigma * (2 - c_sigma))
            h_sigma = 1.0 if norm_p_sigma < threshold else 0.0
            
            # Sigma update
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / np.sqrt(2 * n) - 1))
            
            # Clip sigma to avoid explosion or collapse
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Check for stagnation: if sigma is very small or no improvement for many gens
            # If sigma < 1e-4, restart
            if sigma < 1e-4:
                # Restart: reset mean to best_x, sigma to 1.0, C to I
                mean = best_x.copy()
                sigma = 1.0
                C = np.eye(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                # Maybe add a small random perturbation to mean
                mean += rng.normal(0, 0.1, n)
                mean = np.clip(mean, -5, 5)

        # Final polish with Nelder-Mead if budget remains
        remaining = budget - calls_used
        if remaining > 10 and not check_deadline():
            from scipy.optimize import minimize
            
            def objective(x):
                nonlocal calls_used
                if calls_used >= budget:
                    raise ValueError("Budget exceeded")
                val = func(np.clip(x, -5, 5))
                calls_used += 1
                return val
            
            try:
                res = minimize(
                    objective,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': remaining, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
