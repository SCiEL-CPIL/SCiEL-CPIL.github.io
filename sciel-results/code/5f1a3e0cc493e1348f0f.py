import numpy as np

class Optimizer:
    """CMA-ES continuous optimizer for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        # Standard population size
        lambda_pop = 4 * int(np.floor(np.log2(dim))) + 4
        if lambda_pop < 4:
            lambda_pop = 4
            
        # Number of weighted points
        mu = lambda_pop // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        
        # Variance-effective population size
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial mean and sigma
        x_mean = np.zeros(dim)
        sigma = 1.0
        
        # State variables
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute sqrt of C for sampling
        # We will update this every step
        
        best_x = x_mean.copy()
        best_f = float('inf')
        
        calls = 0
        
        # Helper to update best
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
        # Main loop
        while calls < budget:
            # Generate population
            # Use current C and sigma
            # We need to sample from N(x_mean, sigma^2 * C)
            # Using Cholesky decomposition or eigendecomposition
            # For small dim, eigendecomposition is robust
            
            # Update sqrtC
            # C is symmetric positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-20)
            sqrtC = eigvecs @ np.diag(np.sqrt(eigvals))
            
            # Sample
            population = []
            fvals = []
            
            for i in range(lambda_pop):
                if calls >= budget:
                    break
                
                z = rng.standard_normal(dim)
                x = x_mean + sigma * (sqrtC @ z)
                x = np.clip(x, -5, 5)
                
                f = func(x)
                calls += 1
                
                population.append(x)
                fvals.append(f)
                
                update_best(x, f)
            
            if len(population) < mu:
                break
                
            # Sort population by fitness
            # Only keep top mu for recombination, but we need to handle the rest for adaptation?
            # Standard CMA-ES uses the top mu for recombination weights.
            # The remaining individuals are discarded for recombination but might be used for other adaptations?
            # Usually, we just sort the whole population and take the top mu.
            
            # Sort indices
            indices = np.argsort(fvals)
            # Top mu
            top_indices = indices[:mu]
            
            # Recombination
            y = np.zeros((mu, dim))
            for i, idx in enumerate(top_indices):
                y[i] = population[idx]
                
            # Weighted mean of y
            x_new = x_mean + sigma * (weights @ y)
            x_new = np.clip(x_new, -5, 5)
            
            # Update sigma
            # Calculate distance in normalized space
            y_sigma = (x_new - x_mean) / sigma
            # Project onto C^{-1/2}
            # C^{-1/2} = sqrtC^{-1}
            # Since sqrtC = V * D^0.5, sqrtC^{-1} = D^{-0.5} * V^T
            inv_sqrtC = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
            
            y_sigma_norm = inv_sqrtC @ y_sigma
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma_norm
            
            # Hessian of log det C
            # ||p_sigma||^2
            psq = np.dot(p_sigma, p_sigma)
            
            # Threshold
            hsig = 1.0 if psq / (1 - (1 - c_sigma)**(2 * (calls // lambda_pop + 1))) < (1 + 2 / (dim + 1)) * (1 - c_sigma) else 0.0
            
            # Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.sqrt(psq) / np.sqrt(mu_eff) - 1))
            # Wait, standard formula: sigma *= exp(c_sigma/c_sigma * (||p_sigma||/sqrt(mu_eff) - 1))
            # Actually: sigma *= exp(c_sigma/c_sigma * (||p_sigma||/sqrt(mu_eff) - 1)) is not right.
            # Formula: sigma = sigma * exp(c_sigma/c_sigma * (||p_sigma||/sqrt(mu_eff) - 1))
            # Let's use the standard:
            sigma *= np.exp(c_sigma / c_sigma * (np.sqrt(psq) / np.sqrt(mu_eff) - 1))
            
            # Update C
            # p_c update
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma_norm
            
            # Rank-1 update
            rank1 = np.outer(p_c, p_c)
            
            # Rank-mu update
            # y is the top mu points
            # We need to transform y to standard normal space? No, y is already in the "x_mean + sigma * ..." space relative to old mean?
            # The rank-mu update uses the difference between the sampled points and the new mean?
            # Standard CMA-ES uses the points that were selected for recombination.
            # Let's use the standard formula:
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - hsig) * c_c * (2 - c_c) * C) + c_mu * sum(weights_i * (y_i - x_new) (y_i - x_new)^T) / sigma^2
            
            # Note: The term (y_i - x_new) should be in the "standard normal" space scaled by sigma?
            # Actually, the covariance matrix C represents the shape in the "z" space where x = x_mean + sigma * sqrtC * z.
            # The differences should be transformed back to z-space.
            
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                # y[i] is in x-space.
                # z_i = (y[i] - x_mean) / sigma
                z_i = (y[i] - x_mean) / sigma
                # Transform to standard normal space using inv_sqrtC
                z_i_std = inv_sqrtC @ z_i
                
                # The rank-mu update typically uses the deviations from the new mean in the standard normal space?
                # Or just the weighted outer products of the transformed points?
                # Let's stick to the standard implementation:
                # The points used are the top mu.
                # The update is:
                # C += c_mu * sum(weights_i * (z_i - z_new) (z_i - z_new)^T)
                # Where z_new = (x_new - x_mean) / sigma transformed?
                # Actually, a simpler and robust version uses the raw differences in the "z" space relative to the previous mean?
                # Let's use the formulation from Hansen's paper:
                # C_{t+1} = (1 - c1 - cmu) C_t + c1 (pc pc^T + (1 - hs) cc (2-cc) C_t) + cmu * sum(weights_i * (y_i - x_{t+1}) (y_i - x_{t+1})^T)
                # Here y_i are the transformed points?
                # In many implementations, the "y" vectors are the points in the "standard normal" space.
                
                # Let's define y_std as the points in standard normal space.
                # y_std_i = inv_sqrtC @ ((x_i - x_mean) / sigma)
                # Then the rank-mu part is sum(weights_i * (y_std_i - y_std_new) (y_std_i - y_std_new)^T)
                # Where y_std_new = inv_sqrtC @ ((x_new - x_mean) / sigma)
                
                pass # Placeholder, will calculate below
            
            # Calculate y_std_new
            y_std_new = inv_sqrtC @ ((x_new - x_mean) / sigma)
            
            for i in range(mu):
                z_i = (population[top_indices[i]] - x_mean) / sigma
                z_i_std = inv_sqrtC @ z_i
                diff = z_i_std - y_std_new
                rank_mu += weights[i] * np.outer(diff, diff)
                
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (rank1 + (1 - hsig) * c_c * (2 - c_c) * C) + c_mu * rank_mu
            
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = x_new
            
            # Check for stagnation or singular C
            if np.any(eigvals < 1e-10):
                # Restart or increase sigma
                sigma *= 1.5
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                x_mean = best_x.copy()

        return best_x, best_f
