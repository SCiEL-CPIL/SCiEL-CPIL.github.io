"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls_made = 0
        max_calls = self.budget

        def elapsed_ok():
            return (time.time() - start_time) < deadline and calls_made < max_calls

        # Initial random exploration to find a decent starting point
        # CMA-ES benefits from a good initial mean, so we do a small random search first
        n_init = 10
        for _ in range(n_init):
            if not elapsed_ok():
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            x = np.zeros(self.dim)
            fx = func(x)
            calls_made += 1
            best_fx = fx
            best_x = x.copy()

        # CMA-ES Parameters
        # Population size lambda
        lam = 4
        # Number of selected points mu
        mu = max(1, lam // 2)
        
        # Initial step size sigma
        # Start with a moderate step size, CMA-ES will adapt it
        sigma = 1.0
        
        # Mean vector
        mean = best_x.copy()
        
        # Covariance matrix C (start with identity)
        C = np.eye(self.dim)
        # Inverse covariance matrix
        C_inv = np.eye(self.dim)
        
        # Eigen decomposition cache for efficiency
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # Step size adaptation parameters
        # 1/5 rule or standard CMA-ES adaptation
        # We use a simple adaptation rule: if success rate is high, increase sigma, else decrease
        # Standard CMA-ES uses cumulative path and step-size control, but for 5D and 500 calls,
        # a simplified version works well.
        
        # Path parameters for step size adaptation (simplified)
        # h_sigma = 1 (no damping for simplicity in low dim)
        # We will use a simple success-based adaptation to keep it robust
        # Actually, let's implement a standard-ish CMA-ES step size adaptation
        # rho_sigma = 1 / (sqrt(lam/2) + 1)
        rho_sigma = 1.0 / (np.sqrt(lam / 2.0) + 1.0)
        cs = (mu + 2) / (lam + mu + 5)  # step size adaptation speed
        ds = 1 + 2 * max(0, cs - 1 / lam) + cs  # damping
        
        # Cumulative path for step size
        p_sigma = np.zeros(self.dim)
        
        # Cumulative paths for covariance matrix adaptation
        p_c = np.zeros(self.dim)
        # Update rates
        cmu = min(0.1, 2 / (self.dim * self.dim + 1))  # update rate for covariance
        cc = (4 + mu / lam) / (lam + 4 + 2 * mu / lam)  # update rate for path p_c
        
        # Weights for selected points
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1 / np.sum(w * w)  # effective population size
        
        # Precompute weights for covariance update
        # w_mu = w[:mu]
        
        # Main CMA-ES Loop
        while elapsed_ok():
            # Sample candidates
            candidates = []
            fvals = []
            
            for i in range(lam):
                if not elapsed_ok():
                    break
                
                # Sample from N(mean, sigma^2 * C)
                # Use eigendecomposition for sampling: mean + sigma * eigvecs @ (sqrt(eigvals) * z)
                z = rng.standard_normal(self.dim)
                # Ensure eigvals are positive (numerical stability)
                eigvals_safe = np.maximum(eigvals, 1e-10)
                x_new = mean + sigma * (eigvecs @ (np.sqrt(eigvals_safe) * z))
                
                # Clip to bounds
                x_new = np.clip(x_new, -5, 5)
                
                fx_new = func(x_new)
                calls_made += 1
                
                candidates.append(x_new)
                fvals.append(fx_new)
                
                if fx_new < best_fx:
                    best_fx = fx_new
                    best_x = x_new.copy()
            
            if len(candidates) < lam:
                break
            
            # Sort candidates by fitness
            indices = np.argsort(fvals)
            selected_x = np.array([candidates[i] for i in indices[:mu]])
            selected_f = np.array([fvals[i] for i in indices[:mu]])
            
            # Update mean
            mean_new = np.sum(w[:, np.newaxis] * selected_x, axis=0)
            
            # Update cumulative path for step size
            # z = C^{-1/2} (mean_new - mean) / sigma
            # C^{-1/2} = eigvecs @ diag(1/sqrt(eigvals)) @ eigvecs.T
            # But we have eigvecs and eigvals from C
            # C^{-1/2} (mean_new - mean) = eigvecs @ (diag(1/sqrt(eigvals)) @ (eigvecs.T @ (mean_new - mean)))
            diff = mean_new - mean
            diff_in_eig_space = eigvecs.T @ diff
            diff_scaled = diff_in_eig_space / np.sqrt(eigvals_safe)
            z = eigvecs @ diff_scaled
            z_norm = np.linalg.norm(z)
            
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (z / (sigma * z_norm + 1e-10) if z_norm > 1e-10 else z)
            
            # Update step size
            # chi_n = expected norm of standard normal
            chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Clamp sigma to reasonable range
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Update cumulative path for covariance
            # C^{-1/2} (mean_new - mean) / sigma
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * (z / (sigma * z_norm + 1e-10) if z_norm > 1e-10 else z)
            
            # Update covariance matrix
            # C = (1 - cmu) C + cmu * (sum w_i (x_i - mean)(x_i - mean)^T + h_c p_c p_c^T)
            # h_c = 1 if ||p_c||^2 / (1 - (1-cc)^{2(gen+1)}) < (1+3/dim) * (lam-1+mu) * chi_n^2 else 0
            # For simplicity and stability, we often skip h_c or set it to 1 if p_c is not too large
            # Let's compute h_c
            gen = 0  # Not tracking generations exactly, but we can approximate
            # Actually, let's just use a simple rank-mu update without h_c for robustness in short runs
            # Or use h_c = 1 if ||p_c||^2 < (1 + 3/dim) * (lam - 1 + mu) * chi_n^2 else 0
            pc_norm_sq = np.dot(p_c, p_c)
            h_c = 1.0 if pc_norm_sq < (1 + 3 / self.dim) * (lam - 1 + mu) * chi_n ** 2 else 0.0
            
            # Rank-mu update
            diff_mean = mean_new - mean
            rank_mu = np.sum(w[:, np.newaxis, np.newaxis] * np.einsum('i,ij,ik->jk', 
                                                                    selected_x - mean_new, 
                                                                    selected_x - mean_new, 
                                                                    np.ones((1, 1))), axis=0)
            
            # Actually, simpler way to compute rank-mu:
            # sum w_i (x_i - mean_new)(x_i - mean_new)^T
            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(mu):
                diff_i = selected_x[i] - mean_new
                rank_mu += w[i] * np.outer(diff_i, diff_i)
            
            rank_one = h_c * np.outer(p_c, p_c)
            
            C_new = (1 - cmu * (1 + h_c * (mueff / 2 - 1))) * C + cmu * (rank_mu + rank_one)
            
            # Symmetrize
            C_new = (C_new + C_new.T) / 2
            
            # Update eigen decomposition for next iteration
            # Re-compute eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C_new)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-10)
            
            mean = mean_new
            C = C_new

        return best_x, best_fx
# EVOLVE-BLOCK-END
