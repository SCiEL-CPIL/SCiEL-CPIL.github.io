"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """Bayesian Optimization with Matern-5/2 GP and EI Acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State tracking
        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0
        
        # Helper to evaluate and track best
        def evaluate(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5.0, 5.0)
            if call_count >= budget:
                return np.inf
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Gaussian Process with Matern-5/2 kernel
        def matern52_kernel(X, Y, length_scales, signal_var):
            """
            Compute covariance between X (n x d) and Y (m x d).
            """
            # X: (n, d), Y: (m, d)
            # Compute squared distances
            # X^2 sum over d, Y^2 sum over d
            X_sq = np.sum(X**2, axis=1, keepdims=True) # (n, 1)
            Y_sq = np.sum(Y**2, axis=1, keepdims=True) # (m, 1)
            # (n, m)
            sq_dists = X_sq + Y_sq.T - 2.0 * np.dot(X, Y.T)
            sq_dists = np.maximum(sq_dists, 0.0)
            
            # r = sqrt(sq_dists)
            r = np.sqrt(sq_dists)
            
            # Matern 5/2: (1 + sqrt(3)*r + 3*r^2/3) * exp(-sqrt(3)*r)
            # Actually: (1 + sqrt(3)*r + (sqrt(3)*r)^2/3) * exp(-sqrt(3)*r)
            # Standard form: (1 + sqrt(3)*r + 3*r^2) * exp(-sqrt(3)*r) ? 
            # No, k(r) = (1 + sqrt(3)*r + (sqrt(3)*r)^2/3) * exp(-sqrt(3)*r)
            # Let's use the standard: (1 + sqrt(3)*r + 3*r^2) * exp(-sqrt(3)*r) is for 5/2?
            # Wikipedia: k(r) = (1 + sqrt(3)*r + (sqrt(3)*r)^2/3) * exp(-sqrt(3)*r)
            # Let's simplify: (1 + sqrt(3)*r + r^2) * exp(-sqrt(3)*r) is common approximation?
            # Correct Matern 5/2: (1 + sqrt(3)*r + 3*r^2) * exp(-sqrt(3)*r) is WRONG.
            # Correct: (1 + sqrt(3)*r + (sqrt(3)*r)^2/3) * exp(-sqrt(3)*r)
            # Let's stick to the formula: (1 + sqrt(3)*r + 3*r^2) * exp(-sqrt(3)*r) is actually for something else?
            # Let's use: (1 + sqrt(3)*r + 3*r^2) * exp(-sqrt(3)*r)
            # Wait, (sqrt(3)*r)^2/3 = 3*r^2/3 = r^2.
            # So: (1 + sqrt(3)*r + r^2) * exp(-sqrt(3)*r)
            
            sqrt3_r = np.sqrt(3.0) * r
            kernel_val = (1.0 + sqrt3_r + sqrt3_r**2 / 3.0) * np.exp(-sqrt3_r)
            
            # Apply length scales
            # We assume isotropic length scale for simplicity in 5D to reduce hyperparameter tuning
            # Or anisotropic? Isotropic is safer for BO with limited data.
            # If anisotropic, we need to scale X and Y before distance calc.
            # Let's use isotropic length scale for stability.
            # The `length_scales` parameter in the signature is ignored if we use isotropic, 
            # or we can use a single scalar. Let's use a single scalar length_scale.
            # I will modify the kernel function to take a single length_scale.
            
            # Re-implementing inside this function to be clean
            # Actually, I passed length_scales. Let's assume it's a scalar or array.
            # For simplicity and robustness in 5D, let's use isotropic.
            # I will ignore the `length_scales` array and use a global `length_scale` variable 
            # or just use a fixed one for now to avoid hyperparameter optimization complexity 
            # which might eat into the budget or time.
            # Better: Use a fixed, reasonable length scale for the [-5,5] box.
            # Box width is 10. A length scale of 1.0 or 2.0 is reasonable.
            # Let's use 1.5.
            
            # Wait, the function signature has `length_scales`. I should probably just use a scalar 
            # inside the GP class or closure.
            # Let's rewrite the kernel usage to be simpler.
            
            # I will define the GP parameters as local variables in __call__
            # and use them directly.
            
            return kernel_val * signal_var

        # To avoid confusion, let's define the GP prediction logic inline or with simpler helpers.
        
        # GP Parameters
        # We will use fixed hyperparameters to avoid the cost of MCMC or grid search for them,
        # which is risky with only 500 function calls.
        # Signal variance: The function values are in [-100, 100]. Let's assume std ~ 50.
        # Noise variance: Small, e.g., 1e-6, since the function is deterministic.
        # Length scale: 1.5 (covers a good portion of the 10-unit box).
        
        signal_var = 2500.0  # std = 50
        noise_var = 1e-6
        length_scale = 1.5
        
        # Store observations
        X_obs = []
        y_obs = []
        
        def predict_gp(X_new):
            """
            Predict mean and variance for X_new (m x d) given X_obs (n x d) and y_obs (n).
            Returns mean (m,), var (m,).
            """
            if len(X_obs) == 0:
                # Prior
                return np.zeros(len(X_new)), np.full(len(X_new), signal_var)
            
            X_obs_arr = np.array(X_obs)
            y_obs_arr = np.array(y_obs)
            X_new_arr = np.array(X_new)
            
            # Compute K_obs_obs (n x n)
            # Using isotropic Matern-5/2
            def compute_kernel(A, B):
                # A: (n, d), B: (m, d)
                A_sq = np.sum(A**2, axis=1, keepdims=True)
                B_sq = np.sum(B**2, axis=1, keepdims=True)
                sq_dists = A_sq + B_sq.T - 2.0 * np.dot(A, B.T)
                sq_dists = np.maximum(sq_dists, 0.0)
                r = np.sqrt(sq_dists)
                sqrt3_r = np.sqrt(3.0) * r / length_scale  # Normalize by length scale
                kernel_val = (1.0 + sqrt3_r + sqrt3_r**2 / 3.0) * np.exp(-sqrt3_r)
                return kernel_val * signal_var

            K_obs_obs = compute_kernel(X_obs_arr, X_obs_arr)
            K_obs_new = compute_kernel(X_obs_arr, X_new_arr)
            
            # Add noise to diagonal of K_obs_obs
            K_obs_obs += noise_var * np.eye(len(X_obs_arr))
            
            # Solve K_obs_obs * alpha = y_obs
            try:
                alpha = np.linalg.solve(K_obs_obs, y_obs_arr)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if singular
                alpha = np.linalg.lstsq(K_obs_obs, y_obs_arr, rcond=None)[0]
                
            # Mean
            mu = K_obs_new.T @ alpha
            
            # Variance
            K_new_new_diag = signal_var * np.ones(len(X_new_arr))
            var = K_new_new_diag - np.sum(K_obs_new**2 * alpha, axis=0)
            var = np.maximum(var, 1e-12) # Ensure positive
            
            return mu, var

        def expected_improvement(X, y, mu, var):
            """
            Compute Expected Improvement for minimization.
            """
            sigma = np.sqrt(var)
            # If sigma is 0, EI is 0 if y > best, else 0 (no improvement)
            # Standard formula: EI = (y - mu - xi) * Phi((y - mu - xi) / sigma) + sigma * phi((y - mu - xi) / sigma)
            # For minimization:
            # If y <= mu + xi, EI = 0
            # Let's use xi = 0 for pure exploitation/balance
            xi = 0.0
            
            with np.errstate(divide='ignore', invalid='ignore'):
                z = (y - mu - xi) / sigma
                # Handle sigma = 0
                z = np.where(sigma > 1e-12, z, 0.0)
                
                ei = (y - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
                # If y - mu - xi <= 0, EI should be 0 (or close to it)
                # The formula naturally handles this if we clip negative values?
                # Actually, if z is very negative, cdf is 0, pdf is 0.
                # If y < mu, z is negative.
                # Let's ensure EI is non-negative.
                ei = np.maximum(ei, 0.0)
                
            return ei

        # Phase 1: Initial Random Sampling
        # We need enough points to build a decent GP.
        # In 5D, 10-20 points is a good start.
        n_init = 15
        np.random.seed(42)
        
        for _ in range(n_init):
            if call_count >= budget or time.time() > deadline:
                break
            x = np.random.uniform(-5.0, 5.0, dim)
            f = evaluate(x)
            if f < np.inf:
                X_obs.append(x)
                y_obs.append(f)
        
        # Phase 2: Bayesian Optimization Loop
        # For each remaining call:
        # 1. Generate candidate points
        # 2. Evaluate GP mean/var
        # 3. Calculate EI
        # 4. Pick best EI
        # 5. Evaluate and update
        
        # Number of candidates for EI maximization
        n_candidates = 200
        
        while call_count < budget and time.time() < deadline:
            # Generate candidates
            # Mix of random and local perturbations around best
            candidates = []
            
            # 50% random
            n_rand = n_candidates // 2
            candidates.extend(np.random.uniform(-5.0, 5.0, (n_rand, dim)))
            
            # 50% local perturbations around best_x
            n_local = n_candidates - n_rand
            if best_f < np.inf:
                # Gaussian perturbations with decreasing variance
                # Start with large variance, shrink over time
                progress = call_count / budget
                sigma_local = 2.0 * (1.0 - progress) + 0.2
                local_cands = np.random.normal(best_x, sigma_local, (n_local, dim))
                local_cands = np.clip(local_cands, -5.0, 5.0)
                candidates.extend(local_cands)
            else:
                candidates.extend(np.random.uniform(-5.0, 5.0, (n_local, dim)))
            
            candidates = np.array(candidates)
            
            # Predict
            mu, var = predict_gp(candidates)
            
            # EI
            ei = expected_improvement(candidates, best_f, mu, var)
            
            # Pick best
            idx = np.argmax(ei)
            x_next = candidates[idx]
            
            # Evaluate
            f_next = evaluate(x_next)
            
            if f_next < np.inf:
                X_obs.append(x_next)
                y_obs.append(f_next)
                
        return best_x, best_f

# EVOLVE-BLOCK-END
