"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for Rosenbrock with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        call_count = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        start_time = time.time()
        deadline = start_time + 285.0
        
        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return 1e10
            call_count += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val
        
        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(dim + 1))
        lam = max(4 + int(3 * np.log(dim + 1)), 10)
        
        # Initial step size: sigma = 1.0 (standard for [-5,5]^5)
        sigma = 1.0
        
        # Initial mean: start near origin
        x_mean = np.zeros(dim)
        
        # Covariance matrix: identity
        C = np.eye(dim)
        C_inv = np.eye(dim)
        
        # Eigen-decomposition cache
        eigvals = np.ones(dim)
        eigvecs = np.eye(dim)
        
        # Update counters
        cc = 1.0 / (4 * dim + 4)  # cmc for mean update
        cs = 1.0 / (4 * dim + 4)  # step size adaptation
        c1 = 2 / ((dim + 1)**2 + 10)  # covariance rank-1 update
        cmu = min(0.5 - c1, 2 / ((dim + 2)**2 + 13))  # covariance rank-mu update
        mu = lam // 2  # number of parents
        
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Damping for step size adaptation
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        hs = (1 + 2 * min(1, (mueff - 1) / (dim + mueff)) + c1) ** (1 / (2 * dim * (1 + cs) / chi_n))
        
        # Initial state
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        d_orth = 1.0
        
        def cmaes_step():
            nonlocal x_mean, sigma, C, C_inv, eigvals, eigvecs, p_sigma, p_c, call_count, best_f, best_x
            nonlocal dim, budget, deadline, start_time, wrapped_func, rng
            
            if time.time() > deadline or call_count + lam > budget:
                return False
            
            # Generate population
            # Re-diagonalize if needed (every 1/dim steps or when eigenvalues change significantly)
            # For simplicity, we'll use the cached eigen-decomposition
            # Check if we need to update eigen-decomposition
            if not hasattr(cmaes_step, '_last_eig_update'):
                cmaes_step._last_eig_update = 0
            cmaes_step._last_eig_update += 1
            
            if cmaes_step._last_eig_update % max(1, int(dim / 2)) == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
            
            # Generate samples
            samples = []
            for i in range(lam):
                # Sample from N(0, C) using eigen-decomposition
                z = rng.standard_normal(dim)
                # Transform: eigvecs @ diag(sqrt(sigma * eigvals)) @ z
                x_sample = x_mean + sigma * (eigvecs @ (np.sqrt(eigvals) * z))
                samples.append(x_sample)
            
            # Evaluate all samples
            fvals = []
            for x_sample in samples:
                if call_count >= budget or time.time() > deadline:
                    fvals.append(1e10)
                else:
                    fvals.append(wrapped_func(x_sample))
            
            if call_count >= budget or time.time() > deadline:
                return False
            
            # Sort by fitness
            order = np.argsort(fvals)
            sorted_samples = [samples[i] for i in order]
            sorted_fvals = [fvals[i] for i in order]
            
            # Update mean (recombination)
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += weights[i] * sorted_samples[i]
            
            # Update step size
            d_orth = 1.0  # Simplified: skip d_orth correction for stability
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (C_inv @ (x_new_mean - x_mean) / sigma)
            
            # Check hsig condition
            hsig = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**2) < (1 + 2 / (dim + 1)) * chi_n
            
            if hsig:
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance
            if hsig:
                p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * C_inv @ (x_new_mean - x_mean)
            
            delta_hs = (1 + 2 * min(1, (mueff - 1) / (dim + mueff)) * (1 - hsig)) * hs - 1
            
            rank1 = np.outer(p_c, p_c)
            rankmu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = C_inv @ (sorted_samples[i] - x_mean) / sigma
                rankmu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c1 * delta_hs + c1 * mueff - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
            
            # Update C_inv (approximate via eigen-decomposition)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C_inv = (eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T)
            
            x_mean = x_new_mean
            return True
        
        # Main CMA-ES loop
        while time.time() < deadline and call_count < budget:
            if not cmaes_step():
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
