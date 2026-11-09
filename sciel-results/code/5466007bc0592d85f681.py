import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        x_best = None
        f_best = np.inf
        calls_made = 0
        
        # Phase 1: Global Search via Random Sampling
        # Use a small fraction of the budget to find a good starting point
        # For Discus, finding the valley is key. Random sampling helps locate it.
        n_initial = max(10, budget // 5) # 20% of budget
        n_initial = min(n_initial, budget // 2)
        
        rng = np.random.default_rng(42)
        
        # We will keep the best few points to potentially use as multiple starts if needed,
        # but CMA-ES is a population-based method itself, so starting from the single best
        # is usually sufficient if the global phase is decent.
        
        for i in range(n_initial):
            if time.time() > deadline or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls_made += 1
            if f < f_best:
                f_best = f
                x_best = x.copy()
                
        if x_best is None:
            # Fallback
            x_best = np.zeros(dim)
            f_best = func(x_best)
            calls_made += 1

        # Phase 2: CMA-ES Local Search
        # CMA-ES is highly effective for Discus functions due to covariance adaptation.
        
        # Initial parameters
        sigma = 2.0  # Initial step size, somewhat large to explore, will adapt
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # Standard CMA-ES population size
        
        # Mean vector (start from best found)
        mean = x_best.copy()
        
        # Covariance matrix (start with identity)
        C = np.eye(dim)
        # Eigen decomposition for inverse square root (used in sampling)
        # We'll update C periodically. For simplicity and speed, we can use
        # a simplified update or standard CMA-ES update rules.
        
        # Standard CMA-ES variables
        mu = pop_size // 2  # Number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Helper to sample from multivariate normal with covariance C
        # We need C^{1/2}. We'll recompute eigen decomposition if C changes significantly,
        # but for small dim (5), we can just do it every step or use Cholesky if positive definite.
        # CMA-ES updates C to remain positive definite.
        
        def sample_from_cov(mean, sigma, C, rng):
            # Get square root of C
            # Using eigendecomposition for stability
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues due to numerical error
            eigvals = np.clip(eigvals, 1e-10, None)
            sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            # Sample standard normal
            z = rng.standard_normal(dim)
            x = mean + sigma * (sqrt_C @ z)
            return x

        # Main CMA-ES loop
        while calls_made < budget and time.time() < deadline:
            # Generate population
            x_pop = []
            f_pop = []
            
            for i in range(pop_size):
                if calls_made >= budget or time.time() > deadline:
                    break
                x = sample_from_cov(mean, sigma, C, rng)
                # Clip to box [-5, 5]
                x = np.clip(x, -5, 5)
                f = func(x)
                calls_made += 1
                x_pop.append(x)
                f_pop.append(f)
                
                if f < f_best:
                    f_best = f
                    x_best = x.copy()
            
            if len(x_pop) < 2:
                break
            
            # Sort by fitness
            indices = np.argsort(f_pop)
            x_pop = [x_pop[i] for i in indices]
            f_pop = [f_pop[i] for i in indices]
            
            # Select parents
            x_parents = x_pop[:mu]
            
            # Update mean
            mean_old = mean.copy()
            mean = np.sum([weights[i] * x_parents[i] for i in range(mu)], axis=0)
            
            # Update step size (sigma)
            # p_sigma = (1 - c_sigma) p_sigma + c_sigma * sqrt(mu_eff) * (C^{-1/2} * (mean - mean_old)) / sigma
            # We need C^{-1/2}. We have sqrt_C above.
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            diff = (mean - mean_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (inv_sqrt_C @ diff)
            
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - n_initial)))) / chi_n if calls_made > n_initial else 1.0
            h_sigma = 1.0 if h_sigma < 1.7 else 0.0 # Simplified threshold check
            
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance matrix
            # p_c = (1 - c_c) p_c + h_sigma * c_c * sqrt(mu_eff) * (C^{-1/2} * (mean - mean_old)) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * c_c * np.sqrt(mu_eff) * (inv_sqrt_C @ diff)
            
            h_c = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**(2 * (calls_made - n_initial)))) / chi_n if calls_made > n_initial else 1.0
            h_c = 1.0 if h_c < 1.4 else 0.0
            
            # Rank-1 and Rank-mu update
            # C = (1 + c_1 * h_c * (1 - h_c) - c_1 - c_mu) C + c_1 * p_c * p_c.T + c_mu * sum(weights_i * (x_i - mean_old)(x_i - mean_old).T)
            
            delta_h = (1 - h_c) * c_c * (2 - c_c)
            C = (1 + delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            # Transform parents to standard normal space? No, just use weighted outer products in x-space adjusted by sigma?
            # Standard formula: c_mu / sigma^2 * sum w_i (x_i - mean_old)(x_i - mean_old).T
            # Note: The mean update uses weighted sum, so the scatter is around mean_old.
            
            scatter_sum = np.zeros((dim, dim))
            for i in range(mu):
                diff_parent = x_parents[i] - mean_old
                scatter_sum += weights[i] * np.outer(diff_parent, diff_parent)
                
            C += (c_mu / (sigma * sigma)) * scatter_sum
            
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2.0
            
            # Check for convergence
            if sigma < 1e-15 or (max(f_pop[:mu]) - min(f_pop[:mu])) < 1e-12:
                break

        return x_best, f_best
