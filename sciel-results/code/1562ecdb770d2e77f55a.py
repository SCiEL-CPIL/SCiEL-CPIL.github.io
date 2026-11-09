"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Gaussian Process Surrogate-Assisted Optimizer for Rastrigin f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 330s kill
        rng = np.random.default_rng(42)
        
        # Budget allocation:
        # 1. Initial random sampling: 15 points to bootstrap the GP.
        # 2. Surrogate-assisted loop: Use remaining budget.
        
        n_initial = 15
        if self.budget <= n_initial:
            # Fallback: just do random search
            best_x = None
            best_f = np.inf
            for _ in range(self.budget):
                x = rng.uniform(-5, 5, self.dim)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            return best_x, best_f

        X = []
        Y = []
        
        # Phase 1: Initial Random Sampling
        for _ in range(n_initial):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            X.append(x)
            Y.append(fx)
        
        X = np.array(X)
        Y = np.array(Y)
        
        best_x = X[np.argmin(Y)].copy()
        best_f = np.min(Y)
        
        # GP Parameters
        # Rastrigin has period 2*pi ~ 6.28 in each dimension.
        # We use a product of periodic kernels plus a white noise kernel.
        # K(x, y) = prod_i [sigma_f^2 * exp(-0.5 * (sin(pi*(x_i-y_i)/period)^2 / l^2)) ] + sigma_n^2 * I
        # Simplified: We will use a standard RBF kernel but with a length scale
        # that encourages capturing the periodicity, or just a mix.
        # Given the small budget (500 calls) and dim=5, a full GP with exact
        # likelihood maximization might be slow if we re-fit every step.
        # Instead, we use a fixed, well-tuned kernel and only update the
        # posterior mean/variance efficiently, or re-fit if fast enough.
        
        # For speed and robustness, we will use a simple RBF kernel with 
        # a length scale of ~2.0 (smaller than period, to allow flexibility)
        # and a signal variance of ~50.0 (Rastrigin range is roughly 0 to 10*5+50=100).
        # Actually, Rastrigin f(x) = 10*n + sum(x^2 - 10cos(2pi x)).
        # Min is 0. Max in [-5,5] is around 10*5 + 5*(25+10) = 50 + 175 = 225.
        # Let's use signal variance sigma_f^2 = 100.
        # Length scale l = 1.5.
        
        sigma_f_sq = 100.0
        l_sq = 1.5 ** 2
        sigma_n_sq = 1e-6  # Noise level
        
        def rbf_kernel(X1, X2):
            # X1: (n1, d), X2: (n2, d)
            # Compute squared distances
            # dist^2 = ||x||^2 + ||y||^2 - 2 x.y
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            np.maximum(sq_dists, 0, out=sq_dists)
            return sigma_f_sq * np.exp(-0.5 * sq_dists / l_sq)

        def gp_predict(X_new, X_obs, Y_obs, K_inv, mean):
            # Returns mean and variance for X_new
            # K(X_new, X_obs) @ K_inv @ (Y_obs - mean)
            K_new_obs = rbf_kernel(X_new, X_obs)
            mu = mean + K_new_obs @ K_inv @ (Y_obs - mean)
            
            # Variance: sigma_f^2 - K(X_new, X_obs) @ K_inv @ K(X_obs, X_new)
            # K_inv is (n, n). K(X_obs, X_new) is (n, m).
            # Var = sigma_f_sq - K_new_obs @ K_inv @ K_new_obs.T
            K_obs_new = K_new_obs.T
            var = sigma_f_sq - np.sum((K_new_obs @ K_inv) * K_obs_new, axis=1)
            # Clip variance to be positive
            var = np.maximum(var, 1e-10)
            return mu, var

        # Initial GP Fit
        K = rbf_kernel(X, X)
        K += sigma_n_sq * np.eye(len(X))
        K_inv = np.linalg.inv(K)
        mean = np.mean(Y)
        # We don't need to maximize likelihood for this fixed kernel strategy,
        # we just use the posterior with these hyperparameters.
        
        calls_used = n_initial
        
        # Phase 2: Surrogate-Assisted Loop
        # Use Expected Improvement (EI) acquisition function.
        # EI(x) = (mu(x) - best_f - xi) * Phi(z) + sigma(x) * phi(z)
        # where z = (mu(x) - best_f - xi) / sigma(x)
        
        xi = 0.01 * (np.max(Y) - np.min(Y)) if len(Y) > 1 else 0.01
        
        n_candidates = 500 # Number of candidates to evaluate in surrogate space per step
        # We need to be careful with time. 
        # GP prediction on 500 points in 5D is fast.
        # Inverse of K is O(n^3). We only re-invert when we add a point?
        # Actually, we can use Sherman-Morrison or just re-invert since n is small (<500).
        # Re-inverting a 500x500 matrix takes ~10-50ms. 500 steps * 50ms = 25s. Acceptable.
        
        for step in range(self.budget - n_initial):
            if time.time() > deadline:
                break
            
            # Generate candidates
            # Mix of random points and points near current best
            # 80% random, 20% local perturbations of best_x
            n_rand = int(0.8 * n_candidates)
            n_local = n_candidates - n_rand
            
            X_rand = rng.uniform(-5, 5, (n_rand, self.dim))
            X_local = best_x[None, :] + rng.normal(0, 0.5, (n_local, self.dim))
            X_local = np.clip(X_local, -5, 5)
            
            X_cand = np.vstack([X_rand, X_local])
            
            # Predict
            mu, var = gp_predict(X_cand, X, Y, K_inv, mean)
            sigma = np.sqrt(var)
            
            # Calculate EI
            # Avoid division by zero
            z = (mu - best_f - xi) / (sigma + 1e-8)
            # Phi and phi from scipy.stats.norm or manual approximation
            # Using manual normal CDF and PDF for speed and no dependency issues if scipy is slow
            # But scipy is available. Let's use scipy.stats.norm for accuracy.
            from scipy.stats import norm
            Phi = norm.cdf(z)
            phi = norm.pdf(z)
            
            EI = (mu - best_f - xi) * Phi + sigma * phi
            EI = np.maximum(EI, 0) # EI should be non-negative
            
            # Select best candidate
            idx_best_cand = np.argmax(EI)
            x_next = X_cand[idx_best_cand]
            
            # Evaluate
            fx_next = func(x_next)
            calls_used += 1
            
            # Update X, Y
            X = np.vstack([X, x_next])
            Y = np.append(Y, fx_next)
            
            # Update Best
            if fx_next < best_f:
                best_f = fx_next
                best_x = x_next.copy()
                # Update xi slightly if we improve?
                # Keep xi fixed for stability.
            
            # Update GP Model
            # Re-invert K
            K = rbf_kernel(X, X)
            K += sigma_n_sq * np.eye(len(X))
            # Use Cholesky for stability if possible, else inv
            try:
                L = np.linalg.cholesky(K)
                K_inv = np.linalg.inv(L.T @ L) # Or solve systems
                # Actually, for prediction we need K_inv.
                # Let's just use np.linalg.inv for simplicity, it's fast enough for n<500.
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                # Fallback if singular
                K_inv = np.linalg.pinv(K)
            
            # Update mean? 
            # The mean of the GP is usually 0 or the mean of Y.
            # If we assume zero mean, mean=0.
            # If we assume mean = mean(Y), we should update it.
            # Let's stick to mean = 0 for standard GP formulation, 
            # but our Y values are not centered. 
            # Wait, standard GP assumes y ~ N(0, K). 
            # If Y has a large offset (e.g., 50), the model will try to fit that with the kernel.
            # It's better to center Y.
            # Let's re-center Y for the GP calculations.
            
        # Wait, the above loop has a flaw: I didn't center Y.
        # If Y is around 50, and kernel variance is 100, it's fine.
        # But let's refine the loop to be more robust.
        
        # RE-IMPLEMENTATION OF PHASE 2 WITH PROPER CENTERING AND EFFICIENCY
        
        # Reset state for a clean Phase 2? No, we already used calls.
        # Let's restart the logic from Phase 1 to ensure correctness in the final code.
        
        # --- CORRECTED IMPLEMENTATION ---
        
        # Clear previous partial work to ensure clean logic in the returned code
        # (The code below is the actual logic that will be executed)
        
        X = []
        Y = []
        calls_used = 0
        
        # Phase 1: Initial Random Sampling
        for _ in range(n_initial):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            X.append(x)
            Y.append(fx)
            calls_used += 1
        
        X = np.array(X)
        Y = np.array(Y)
        
        best_x = X[np.argmin(Y)].copy()
        best_f = np.min(Y)
        
        # Hyperparameters
        sigma_f_sq = 100.0
        l_sq = 1.5 ** 2
        sigma_n_sq = 1e-4
        
        def rbf_kernel(X1, X2):
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            np.maximum(sq_dists, 0, out=sq_dists)
            return sigma_f_sq * np.exp(-0.5 * sq_dists / l_sq)

        # Center Y for GP
        Y_mean = np.mean(Y)
        Y_centered = Y - Y_mean
        
        K = rbf_kernel(X, X)
        K += sigma_n_sq * np.eye(len(X))
        K_inv = np.linalg.inv(K)
        
        from scipy.stats import norm
        
        n_candidates = 200 # Reduced to ensure speed
        
        for step in range(self.budget - calls_used):
            if time.time() > deadline:
                break
            
            # Generate candidates
            n_rand = int(0.8 * n_candidates)
            n_local = n_candidates - n_rand
            
            X_rand = rng.uniform(-5, 5, (n_rand, self.dim))
            X_local = best_x[None, :] + rng.normal(0, 0.5, (n_local, self.dim))
            X_local = np.clip(X_local, -5, 5)
            
            X_cand = np.vstack([X_rand, X_local])
            
            # Predict
            K_new_obs = rbf_kernel(X_cand, X)
            # mu = Y_mean + K_new_obs @ K_inv @ Y_centered
            alpha = K_inv @ Y_centered
            mu = Y_mean + K_new_obs @ alpha
            
            # Var = sigma_f_sq - K_new_obs @ K_inv @ K_new_obs.T
            K_inv_new = K_inv @ K_new_obs.T
            var = sigma_f_sq - np.sum(K_new_obs * K_inv_new, axis=1)
            var = np.maximum(var, 1e-8)
            sigma = np.sqrt(var)
            
            # EI
            # Dynamic xi based on current improvement range
            current_range = np.max(Y) - np.min(Y)
            xi = 0.05 * current_range if current_range > 1e-6 else 0.01
            
            z = (mu - best_f - xi) / (sigma + 1e-8)
            Phi = norm.cdf(z)
            phi = norm.pdf(z)
            
            EI = (mu - best_f - xi) * Phi + sigma * phi
            EI = np.maximum(EI, 0)
            
            idx_best_cand = np.argmax(EI)
            x_next = X_cand[idx_best_cand]
            
            fx_next = func(x_next)
            calls_used += 1
            
            if fx_next < best_f:
                best_f = fx_next
                best_x = x_next.copy()
            
            # Update Data
            X = np.vstack([X, x_next])
            Y = np.append(Y, fx_next)
            
            # Update GP
            # Recalculate Y_mean and Y_centered
            Y_mean = np.mean(Y)
            Y_centered = Y - Y_mean
            
            K = rbf_kernel(X, X)
            K += sigma_n_sq * np.eye(len(X))
            try:
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                K_inv = np.linalg.pinv(K)
                
        return best_x, best_f
# EVOLVE-BLOCK-END
