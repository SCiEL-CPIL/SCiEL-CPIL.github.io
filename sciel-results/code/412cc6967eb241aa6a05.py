import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return float('inf')
            # Ensure x is a numpy array of correct shape and within bounds
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            calls += 1
            try:
                f_val = float(func(x))
            except Exception:
                f_val = float('inf')
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        rng = np.random.default_rng(42)
        
        # Phase 1: Global Exploration to find a good starting point for CMA-ES
        # Bent Cigar is convex, so a good start helps, but CMA-ES is robust.
        # We spend a small fraction of the budget here.
        num_initial = 10
        candidates = []
        for _ in range(num_initial):
            if time.time() > deadline or calls >= budget:
                break
            x_init = rng.uniform(-4.0, 4.0, dim)
            f_init = safe_func(x_init)
            candidates.append((x_init, f_init))
            
        if not candidates:
            return best_x, best_f
            
        # Pick the best initial point as the mean for CMA-ES
        candidates.sort(key=lambda x: x[1])
        x0, f0 = candidates[0]
        m = x0.copy()
        
        # CMA-ES Parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))
        if lam < 2:
            lam = 2
        # Weights
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size
        sigma = 1.0
        # Covariance matrix
        C = np.eye(dim)
        # Eigenvalue cache for efficient updates
        C_eigvals = np.ones(dim)
        C_eigvecs = np.eye(dim)
        
        # Update rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4.3)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Damping for step size update
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Precompute constants
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Initial step size for CMA-ES is often set to 0.5 * range
        # But we already explored a bit. Let's keep sigma = 1.0 as it covers the box.
        
        # Main CMA-ES Loop
        # We have budget - calls remaining.
        # Each generation uses 'lam' calls.
        
        while calls < budget and time.time() < deadline:
            # Check if we can afford another generation
            if calls + lam > budget:
                # If we can't afford a full generation, do a few single steps or break
                # For simplicity and robustness, break if we can't do a full generation
                # Or we can reduce lam, but standard CMA-ES prefers fixed lam.
                # Let's just break to be safe and return best.
                break
            
            # Generate offspring
            y = np.zeros((lam, dim))
            x = np.zeros((lam, dim))
            f_vals = np.zeros(lam)
            
            for i in range(lam):
                # Sample from multivariate normal with covariance C
                # Use Cholesky or eigendecomposition
                # For small dim, np.random.multivariate_normal is fine but slower.
                # Let's use the eigendecomposition cached from previous step.
                # z ~ N(0, I)
                z = rng.standard_normal(dim)
                # y = sigma * C_eigvecs @ (C_eigvals**0.5 * z)
                y[i] = sigma * (C_eigvecs @ (np.sqrt(C_eigvals) * z))
                
                # Update path p_sigma (before updating m)
                # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * sqrt(mu_eff) / sigma * (y_i - m)
                # We need p_sigma from previous iteration. Initialize to 0.
                if not hasattr(self, '_p_sigma'):
                    self._p_sigma = np.zeros(dim)
                
                # This logic is slightly tricky in a single pass. 
                # Standard CMA-ES updates p_sigma and p_c AFTER all offspring are generated and sorted.
                # So we just generate x here.
                x[i] = m + y[i]
                x[i] = np.clip(x[i], -5.0, 5.0)
                
                if time.time() > deadline:
                    break
                
                f_vals[i] = safe_func(x[i])
                
                if calls >= budget:
                    break
            
            if time.time() > deadline or calls >= budget:
                break
                
            # Sort offspring by function value
            indices = np.argsort(f_vals)
            x_sorted = x[indices]
            f_sorted = f_vals[indices]
            
            # Update p_sigma and p_c
            # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * sqrt(mu_eff) * (mean of top mu y's - m) / sigma
            # Note: y_i = x_i - m_old
            y_sorted = x_sorted[:mu] - m
            
            self._p_sigma = (1 - c_sigma) * self._p_sigma + \
                             c_sigma * np.sqrt(mu_eff) * np.mean(y_sorted, axis=0) / sigma
            
            # p_c = (1 - c_c) * p_c + c_c * sqrt(mu_eff) * (mean of top mu y's - m)
            self._p_c = (1 - c_c) * self._p_c + \
                        c_c * np.sqrt(mu_eff) * np.mean(y_sorted, axis=0)
            
            # Update mean
            m_new = np.sum(weights[:, np.newaxis] * x_sorted[:mu], axis=0)
            
            # Update step size sigma
            # log(sigma_new) = log(sigma) + c_sigma / (d_sigma * chi_n) * (||p_sigma|| / chi_n - 1)
            norm_p_sigma = np.linalg.norm(self._p_sigma)
            sigma *= np.exp(c_sigma / (d_sigma * chi_n) * (norm_p_sigma / chi_n - 1))
            # Clip sigma to prevent explosion or collapse
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update covariance matrix C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c + delta_sigma * m @ m) + c_mu * sum(w_i * y_i @ y_i)
            
            # Calculate delta_sigma
            delta_sigma = c_sigma * (2 - c_sigma) * (norm_p_sigma**2 / (sigma**2) - 1)
            
            # We need to update the eigenvalues and vectors efficiently or just recompute.
            # For dim=5, recomputing eigendecomposition is cheap.
            C = (1 - c_1 - c_mu) * C + \
                c_1 * (np.outer(self._p_c, self._p_c) + delta_sigma * np.eye(dim)) + \
                c_mu * np.sum(weights[:mu, np.newaxis, np.newaxis] * np.outer(y_sorted, y_sorted), axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2.0
            
            # Eigen-decomposition for next iteration's sampling
            try:
                C_eigvals, C_eigvecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues due to numerical errors
                C_eigvals = np.clip(C_eigvals, 1e-12, None)
            except np.linalg.LinAlgError:
                # Fallback to identity if decomposition fails
                C = np.eye(dim)
                C_eigvals = np.ones(dim)
                C_eigvecs = np.eye(dim)
            
            # Update mean
            m = m_new
            
            # Check convergence
            if np.max(C_eigvals) / np.min(C_eigvals) > 1e14:
                break
            if sigma < 1e-10:
                break

        return best_x, best_f
