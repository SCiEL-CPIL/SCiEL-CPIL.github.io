"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES Optimizer with adaptive population sizing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # Fixed seed for reproducibility within the instance
        # Using a seed derived from the budget/dim to ensure consistency
        # but random enough to be different across different task instances if they existed
        # Here we just use a fixed seed for the run.
        np.random.seed(42)

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5, 5)
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                f_val = np.inf
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # CMA-ES Parameters
        # Initial sigma
        sigma = 2.0 
        # Initial mean
        mean = np.zeros(dim)
        
        # Population size calculation
        # Standard CMA-ES: lambda = 4 + floor(3 * log(n))
        # For n=5, lambda = 4 + floor(3*1.6) = 4 + 4 = 8
        # We will adaptively reduce this to use more budget for later stages
        # if convergence is fast, or keep it high for exploration.
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Learning rates
        mu_eff = 1.0 # We will compute this properly
        # Standard rates
        # c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        # c_c = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
        # c_1 = 2 / ((n + 1.3)^2 + mu_eff)
        # c_mu = min(1-c_1, 2 * ((mu_eff+2)/(n+mu_eff) - 1) / ((n+2)^2 + mu_eff))
        
        def get_params(n, mu):
            # mu is number of parents
            # Usually mu = lambda/2
            c_sigma = (mu + 2) / (n + mu + 5)
            c_c = (4 + mu/n) / (n + 4 + 2*mu/n)
            c_1 = 2 / ((n + 1.3)**2 + mu)
            c_mu = min(1 - c_1, 2 * ((mu + 2)/(n + mu) - 1) / ((n + 2)**2 + mu))
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            return c_sigma, c_c, c_1, c_mu, chi_n

        # Initial lambda
        lambda_ = 4 + int(3 * np.log(dim))
        if lambda_ < 4: lambda_ = 4
        if lambda_ > 20: lambda_ = 20 # Cap for efficiency
        
        mu = max(1, lambda_ // 2)
        # Weights for recombination
        # Log-linear weights
        w_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w_raw / np.sum(w_raw)
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma, c_c, c_1, c_mu, chi_n = get_params(dim, mu_eff)

        # Generate initial population
        def generate_population(mean, sigma, C, lambda_):
            # Sample from N(0, C)
            # Use Cholesky decomposition for C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset
                C = np.eye(dim)
                L = np.eye(dim)
            
            z = np.random.randn(lambda_, dim)
            # y = L * z^T
            y = z @ L.T
            return mean + sigma * y

        # Main Loop
        # We will run CMA-ES until budget is exhausted or time is up.
        # To handle the "early progress" scoring, we start with a reasonable sigma.
        
        # Optional: Initial quick check to see if we are near the optimum?
        # CMA-ES is good at this.
        
        # Adaptive Lambda: 
        # If the step size sigma becomes very small, we can reduce lambda to save calls
        # or increase it if we are stuck? 
        # Actually, reducing lambda reduces diversity. 
        # A common trick: if sigma < 1e-10, stop.
        
        # We will run for a fixed number of generations or until budget runs out.
        # Each generation costs lambda_ calls.
        # Total budget 500.
        # If lambda=8, we can do ~60 generations.
        
        while calls_made < budget - 10 and time.time() < deadline:
            # 1. Sample population
            pop = generate_population(mean, sigma, C, lambda_)
            f_vals = np.array([safe_call(x) for x in pop])
            
            # 2. Sort by fitness
            idx = np.argsort(f_vals)
            pop = pop[idx]
            f_vals = f_vals[idx]
            
            # 3. Update mean
            # Weighted mean of the best mu individuals
            mean = np.sum(pop[:mu] * w[:, np.newaxis], axis=0)
            
            # 4. Update covariance matrix and step size
            # Standard CMA-ES update rules
            
            # Calculate step size update
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * L^T * (mean - old_mean) / sigma
            # This requires storing old_mean. 
            # Let's simplify by using a slightly more robust/standard implementation structure.
            
            # Actually, let's stick to a simpler variant if the full CMA-ES is too complex to implement correctly in a short block without bugs.
            # However, CMA-ES is the gold standard. Let's implement the standard update carefully.
            
            # We need old_mean for p_sigma update.
            # Let's restructure to keep state.
            
            # Re-initializing state for the loop to be cleaner
            pass # Placeholder

        # The above loop sketch had a state management issue (old_mean). 
        # Let's rewrite the core loop properly.
        
        # Reset state for the actual run
        mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # We'll use a while loop that tracks state
        old_mean = mean.copy()
        
        # Pre-compute L for sampling
        # We need to update C every step, so L changes.
        
        while calls_made < budget - 10 and time.time() < deadline:
            # 1. Sample
            try:
                L = np.linalg.cholesky(C)
            except:
                C = np.eye(dim)
                L = np.eye(dim)
            
            z = np.random.randn(lambda_, dim)
            y = z @ L.T
            pop = mean + sigma * y
            
            # 2. Evaluate
            f_vals = np.array([safe_call(x) for x in pop])
            
            # 3. Sort
            idx = np.argsort(f_vals)
            pop = pop[idx]
            f_vals = f_vals[idx]
            
            # 4. Update Mean
            new_mean = np.sum(pop[:mu] * w[:, np.newaxis], axis=0)
            
            # 5. Update Paths and Sigma
            # p_sigma update
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * L^T * (new_mean - old_mean) / sigma
            if sigma > 1e-10:
                diff_mean = (new_mean - old_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (L.T @ diff_mean)
                
                # Hessianian norm check
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made // lambda_))) ) / chi_n - 1
                h_sigma = 1 if h_sigma < 2.3 / dim else 0 # Approximation for h_sigma indicator
            
            # p_c update
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / chi_n * (L.T @ ((new_mean - old_mean) / sigma))
            
            # Sigma update
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Covariance update
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c.T + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(w_i * y_i * y_i.T)
            
            # Calculate y_i for parents
            # y_i = (pop_i - old_mean) / sigma
            parents_y = (pop[:mu] - old_mean) / sigma if sigma > 1e-10 else np.zeros((mu, dim))
            
            # Rank-one update
            rank_one = p_c * p_c.T + (1 - h_sigma) * c_c * (2 - c_c) * C
            
            # Rank-mu update
            # sum(w_i * y_i * y_i.T)
            # Efficient calculation
            Y = parents_y
            W_sqrt = np.sqrt(w)
            YW = Y * W_sqrt[:, np.newaxis]
            rank_mu = YW.T @ YW
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure positive semi-definiteness
            # Symmetrize
            C = (C + C.T) / 2.0
            
            # Check condition number or eigenvalues
            # If C becomes ill-conditioned, reset or regularize
            # Simple regularization: add small identity if needed
            # For now, rely on Cholesky check in next iteration
            
            old_mean = new_mean
            mean = new_mean
            
            # Optional: Adapt Lambda
            # If sigma is very small, we can reduce lambda to save budget for more generations?
            # Or if we are stuck, increase it?
            # Let's keep it simple: if sigma < 0.1, reduce lambda to 4 to save calls.
            # If sigma > 1.0, keep lambda.
            if sigma < 0.5 and lambda_ > 4:
                lambda_ = max(4, lambda_ // 2)
                mu = max(1, lambda_ // 2)
                # Recalculate weights and rates
                w_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w = w_raw / np.sum(w_raw)
                mu_eff = 1.0 / np.sum(w**2)
                c_sigma, c_c, c_1, c_mu, chi_n = get_params(dim, mu_eff)
            elif sigma > 1.0 and lambda_ < 8:
                lambda_ = min(8, lambda_ * 2)
                mu = max(1, lambda_ // 2)
                w_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w = w_raw / np.sum(w_raw)
                mu_eff = 1.0 / np.sum(w**2)
                c_sigma, c_c, c_1, c_mu, chi_n = get_params(dim, mu_eff)

            # Termination criteria
            if sigma < 1e-10:
                break
            if np.allclose(pop[0], pop[-1], atol=1e-10):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
