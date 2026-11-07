"""CMA-ES optimizer for BBOB f20 (Schwefel) with explicit deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f20 (Schwefel).
    Uses a population-based approach to explore the rotated and shifted landscape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # CMA-ES Parameters
        # Standard CMA-ES for dim=5
        n = self.dim
        # Population size: lambda = 4 + floor(3 * ln(n))
        # For n=5, ln(5) ~ 1.6, 3*1.6 = 4.8, floor = 4. lambda = 8.
        # Let's use a slightly larger population for better exploration in early stages.
        # lambda = 12 is a good balance for n=5.
        pop_size = 12
        mu = pop_size // 2  # Number of selected parents
        
        # Initial mean: Center of the domain
        x_mean = np.zeros(n)
        
        # Initial step size: 
        # The domain is [-5, 5]. The global minimum of Schwefel is at x_i = 4.701041...
        # However, the instance is shifted/rotated. 
        # A reasonable initial sigma is 1.0 or 2.0. 
        # Given the box size is 10, sigma=2.0 covers a good portion.
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(n)
        
        # Step size adaptation parameters (1/5 success rule)
        c_sigma = 1 / (n + 4)
        d_sigma = 1 + 2 * max(0, sqrt(n) - 1) / (n + 3)
        # Wait, d_sigma formula is often 1 + 2*max(0, sqrt((mu_eff-1)/(n+1))-1) + c_sigma
        # Let's use standard values for robustness
        # c_sigma = 1/(n+4)
        # d_sigma = 1 + 2*max(0, sqrt((mu-1)/(n+1))-1) + c_sigma
        mu_eff = mu  # Approximation for simplicity, or calculate properly
        # Proper mu_eff = sum(c_i)^2 / sum(c_i^2)
        # For equal weights w_i = 1/mu, mu_eff = mu/2
        # Let's stick to simple robust values
        c_sigma = 1 / (n + 4)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation parameters
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_c = 1 / (n * n + 6)  # Robust value
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Evolution path
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Weights
        w = np.ones(mu) * (1.0 / mu)
        # Standard weights are often proportional to log ranks, but equal weights are simpler and robust enough for this size
        # Let's use standard CMA-ES weights: w_i = log((mu+1)/2) - log(i)
        # Normalized
        w_raw = np.array([np.log((mu + 1) / 2) - np.log(i) for i in range(1, mu + 1)])
        w = w_raw / np.sum(w_raw)
        
        # Call count
        call_count = 0
        
        # Initial evaluation
        # We need to evaluate the initial mean first? 
        # Usually CMA-ES evaluates the population. 
        # To save calls and get an early best, let's evaluate x_mean first.
        fx_mean = func(x_mean)
        call_count += 1
        update_best(x_mean, fx_mean)
        
        # Main loop
        while call_count < self.budget:
            if time.time() > deadline:
                break
            
            # Check if we have enough budget for a full generation
            if call_count + pop_size > self.budget:
                # Do a partial generation or single step?
                # To avoid wasting calls, let's do a single (1+1)-ES step for the remainder
                # This ensures we use all budget effectively
                for _ in range(self.budget - call_count):
                    if time.time() > deadline:
                        break
                    # Simple adaptive step from current mean
                    y = np.clip(x_mean + sigma * rng.standard_normal(n), -5, 5)
                    fy = func(y)
                    call_count += 1
                    if fy < fx_mean:
                        x_mean = y
                        fx_mean = fy
                        update_best(x_mean, fx_mean)
                        sigma *= 1.1
                    else:
                        sigma *= 0.9
                    sigma = np.clip(sigma, 0.01, 5.0)
                break
            
            # Generate population
            # Cholesky factor of C
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to identity if C is not positive definite (should be rare)
                B = np.eye(n)
                C = np.eye(n)
            
            # Generate normal samples
            Y = rng.standard_normal((pop_size, n))
            # Transform to X
            X = x_mean[None, :] + sigma * (Y @ B.T)
            
            # Clip to bounds
            X = np.clip(X, -5, 5)
            
            # Evaluate population
            F = np.zeros(pop_size)
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                F[i] = func(X[i])
                call_count += 1
                update_best(X[i], F[i])
                
                # If we hit the deadline mid-population, we still have partial info
                # But CMA-ES needs the full population. 
                # If we break here, we should exit the main loop to avoid inconsistency.
                if call_count >= self.budget:
                    break
                if time.time() > deadline:
                    break
            
            # If we didn't complete the population, break out
            if call_count >= self.budget or time.time() > deadline:
                break
            
            # Sort by fitness
            idx = np.argsort(F)
            # Take the best mu
            X_best = X[idx[:mu]]
            F_best = F[idx[:mu]]
            
            # Update mean
            x_mean_new = np.sum(w[:, None] * X_best, axis=0)
            
            # Update covariance matrix
            # Diff
            diff = X_best - x_mean[None, :]
            
            # Rank-1 update (from p_c)
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) / np.linalg.norm(x_mean_new - x_mean) * (x_mean_new - x_mean)
            # Note: If norm is 0, skip update or handle carefully
            if np.linalg.norm(x_mean_new - x_mean) > 1e-12:
                C = (1 + c_c * (2 - c_c) - c_1) * C + c_1 * np.outer(p_c, p_c)
            else:
                C = (1 - c_1) * C
            
            # Rank-mu update
            C = (1 - c_mu) * C + c_mu * np.sum(w[:, None] * (diff @ diff.T), axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update sigma
            # Evolution path for sigma
            p_sigma = (1 - c_sigma) * p_sigma + (c_sigma * d_sigma * np.linalg.norm(x_mean_new - x_mean) / sigma - (1 - c_sigma)) * (1 - c_sigma)
            # Wait, the formula is:
            # p_sigma = (1-c_sigma)*p_sigma + c_sigma*d_sigma * (||x_new - x_old|| / sigma - (1-c_sigma))
            # Let's use the standard update:
            norm_diff = np.linalg.norm(x_mean_new - x_mean)
            p_sigma = (1 - c_sigma) * p_sigma + (c_sigma * d_sigma * (norm_diff / sigma - (1 - c_sigma)))
            
            # Hessian potential for chi-normal
            # chi_n = sqrt(n) * (1 + 1/(2n) - 1/(4n^2))
            chi_n = np.sqrt(n) * (1 + 1.0 / (2 * n) - 1.0 / (4 * n ** 2))
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Clip sigma
            sigma = np.clip(sigma, 0.01, 10.0)
            
            # Update mean
            x_mean = x_mean_new
            
            # Check for termination conditions
            # 1. Small spread
            # 2. Small step size
            # 3. Large step size (diverging)
            # 4. Ill-conditioned C
            # 5. No improvement in last X generations (stagnation)
            
            # Simple stagnation check: if best_f hasn't improved in last 20 generations
            # We don't track history easily here, so we rely on CMA-ES properties.
            # If sigma becomes very small, we are likely stuck.
            if sigma < 0.01:
                break
                
            # If the spread of the population is very small
            # This is implicitly handled by sigma shrinking
            
            # If we are stuck (no improvement), CMA-ES might take a long time to restart.
            # For a fixed budget of 500, if we are stuck early, we might waste calls.
            # However, CMA-ES is robust. 
            # Let's add a simple restart mechanism if we detect extreme stagnation.
            # If no improvement in last 50 evaluations (approx 4 generations), restart with a new random center?
            # This is complex to track. Let's trust CMA-ES for now, but monitor.
            
            # Optional: If C becomes ill-conditioned
            try:
                eigvals = np.linalg.eigvalsh(C)
                if eigvals.max() / max(eigvals.min(), 1e-12) > 1e12:
                    break
            except:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
