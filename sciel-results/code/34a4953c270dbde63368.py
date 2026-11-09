"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Sparse Gaussian Process surrogate optimizer for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf

        # Hyperparameters for the GP
        # We use a simple Matern 5/2 kernel, which is suitable for smooth-ish but non-analytic functions like Rastrigin
        # Length scales are initialized heuristically
        # Since the domain is [-5, 5]^5, a length scale of 1.0 is a reasonable start.
        # Signal variance (sigma_f^2) and noise variance (sigma_n^2)
        # We normalize the function values to improve numerical stability
        
        # State variables for GP
        X_obs = []  # List of observed points
        y_obs = []  # List of observed values (normalized)
        
        # Kernel parameters
        # We will fix the kernel parameters to avoid expensive hyperparameter optimization,
        # which would eat into the function evaluation budget.
        # However, we might adapt length scales coarsely if we had time, but for now fixed.
        # Lunacek has features at multiple scales. A mix of scales might be needed.
        # Let's use a single Matern 5/2 with length scale 0.5 to capture local oscillations 
        # and a large signal variance.
        length_scales = np.ones(dim) * 0.5
        sigma_f = 1.0
        sigma_n = 1e-6
        
        def matern_52_kernel(X1, X2, ls, sf):
            """
            Compute the Matern 5/2 kernel matrix between X1 and X2.
            X1: (n1, dim), X2: (n2, dim)
            """
            # Compute Euclidean distances
            # (n1, 1, dim) * (1, 1, dim) -> (n1, 1, dim)
            # (n1, 1, dim) - (1, n2, dim) -> (n1, n2, dim)
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=2)
            
            # Matern 5/2
            sqrt_5 = np.sqrt(5.0)
            r = sqrt_5 * np.sqrt(sq_dist)
            
            K = (1.0 + r + r**2 / 3.0) * np.exp(-r)
            K *= sf**2
            
            return K

        def update_gp(x_new, y_new):
            """Update the GP state with a new observation."""
            nonlocal X_obs, y_obs
            X_obs.append(x_new.copy())
            y_obs.append(y_new)
            
            if len(X_obs) > 1:
                X_obs = np.array(X_obs)
                y_obs = np.array(y_obs)
            else:
                X_obs = np.array([x_new])
                y_obs = np.array([y_new])

        def predict_gp(X_pred):
            """
            Predict mean and variance at X_pred using the current GP.
            Returns: mean (n, 1), var (n, 1)
            """
            if len(X_obs) == 0:
                return np.zeros((len(X_pred), 1)), np.ones((len(X_pred), 1)) * sigma_f**2
            
            X_train = X_obs
            y_train = y_obs
            
            K = matern_52_kernel(X_train, X_train, length_scales, sigma_f)
            K += np.eye(len(X_train)) * sigma_n
            
            K_star = matern_52_kernel(X_pred, X_train, length_scales, sigma_f)
            
            try:
                # Solve K * alpha = y
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                
                mean = K_star @ alpha
                v = np.linalg.solve(L, K_star.T)
                var = sigma_f**2 - np.sum(v**2, axis=0)
                var = np.maximum(var, 1e-8) # Ensure positive variance
            except np.linalg.LinAlgError:
                # Fallback to least squares if Cholesky fails
                try:
                    alpha = np.linalg.lstsq(K, y_train, rcond=None)[0]
                    mean = K_star @ alpha
                    var = np.ones((len(X_pred), 1)) * sigma_f**2 # Rough estimate
                except:
                    return np.zeros((len(X_pred), 1)), np.ones((len(X_pred), 1)) * sigma_f**2
            
            return mean, var

        def expected_improvement(mean, var, best_y):
            """
            Compute Expected Improvement.
            mean, var: arrays of shape (n, 1)
            best_y: scalar
            """
            sigma = np.sqrt(var)
            z = (best_y - mean) / sigma
            # Standard normal CDF and PDF
            # We can use scipy.stats.norm if available, but let's implement simple approximations
            # or use numpy's erf
            cdf = 0.5 * (1 + np.erf(z / np.sqrt(2)))
            pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
            
            ei = (best_y - mean) * cdf + sigma * pdf
            # For minimization, EI is usually (f_best - f) * Phi(z) + sigma * phi(z)
            # where z = (f_best - f) / sigma
            # This matches the formula above.
            return ei

        # Initial points to seed the GP
        # We need a diverse set to start the surrogate.
        # Use a small Sobol-like or grid set.
        # Given dim=5, a grid of 2^5=32 points is feasible and provides good coverage.
        # But 32 calls is 6.4% of budget. Let's use 16 points: center + corners?
        # Or just 5 axis-aligned points + center + 4 random?
        # Let's use a deterministic set of 10 points for initialization.
        
        init_points = [
            np.zeros(dim),
            np.array([4.0, 4.0, 4.0, 4.0, 4.0]),
            np.array([-4.0, -4.0, -4.0, -4.0, -4.0]),
            np.array([4.0, -4.0, 4.0, -4.0, 4.0]),
            np.array([-4.0, 4.0, -4.0, 4.0, -4.0]),
            np.array([5.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([-5.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 5.0, 0.0, 0.0, 0.0]),
            np.array([0.0, -5.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 5.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -5.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 5.0, 0.0]),
            np.array([0.0, 0.0, 0.0, -5.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 5.0]),
            np.array([0.0, 0.0, 0.0, 0.0, -5.0]),
        ]
        
        # Normalize y values for GP stability
        y_min = np.inf
        y_max = -np.inf
        y_norm_offset = 0.0
        y_norm_scale = 1.0
        y_norm_initialized = False
        
        def normalize_y(y):
            nonlocal y_min, y_max, y_norm_offset, y_norm_scale, y_norm_initialized
            if not y_norm_initialized:
                y_min = y
                y_max = y
                y_norm_initialized = True
                return 0.0 # Map first point to 0
            
            if y < y_min:
                y_min = y
            if y > y_max:
                y_max = y
            
            # Simple scaling to [-1, 1] roughly, but we need to update existing points?
            # No, that's complex. Let's just subtract the running min and divide by a running range estimate.
            # Actually, for GP, affine transformations are fine.
            # Let's use a fixed scaling based on the first few points or just raw values if they are stable.
            # Lunacek values can be large. Let's just use raw values but add a large constant to avoid negative issues?
            # GP handles negative values fine.
            # The main issue is numerical conditioning.
            # Let's stick to raw values but ensure K is well-conditioned.
            # If values are huge, K might be ill-conditioned.
            # Let's normalize to mean 0, var 1 online.
            # For simplicity in this implementation, I will just return the raw value.
            # The kernel variance sigma_f^2 should be large enough to cover the range.
            # Let's set sigma_f based on the initial spread.
            return y

        # Evaluate initial points
        for p in init_points:
            if calls_made >= budget or time.time() > deadline:
                break
            fx = func(p)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = p.copy()
            
            # Add to GP
            X_obs.append(p.copy())
            y_obs.append(fx)
            
            # Update sigma_f estimate
            if len(y_obs) == 1:
                sigma_f = 1.0
            else:
                # Estimate variance of observed y
                y_arr = np.array(y_obs)
                var_y = np.var(y_arr)
                # Set sigma_f to cover the variance
                sigma_f = max(1.0, np.sqrt(var_y) * 2)
                
        if len(X_obs) > 0:
            X_obs = np.array(X_obs)
            y_obs = np.array(y_obs)

        # Main loop: Bayesian Optimization
        # We will generate a batch of candidate points, evaluate EI, and pick the best.
        # To keep it efficient, we use a local search around the current best and random points.
        
        # Candidate generation strategy:
        # 1. 50% Random points in [-5, 5]^5
        # 2. 50% Points sampled from a Gaussian around the current best_x (if exists)
        
        n_candidates = 100 # Number of candidates to evaluate EI for
        
        while calls_made < budget and time.time() < deadline:
            # Generate candidates
            candidates = []
            
            # Random candidates
            n_rand = n_candidates // 2
            # Use deterministic pseudo-random for reproducibility? 
            # The task says "hidden seeded shift", implying the function is fixed per instance.
            # We can use a fixed seed for our internal RNG to ensure deterministic behavior across runs?
            # No, the evaluator creates a new Optimizer for each instance.
            # Using a fixed seed inside __call__ is fine.
            rng = np.random.default_rng(42) # Fixed seed for internal candidates
            
            rand_pts = rng.uniform(-5, 5, size=(n_rand, dim))
            candidates.extend(rand_pts)
            
            # Local candidates around best
            if best_x is not None:
                n_local = n_candidates - n_rand
                # Adaptive sigma for local search
                # Start large, shrink as we progress?
                # Let's use a fixed set of scales for local search
                local_scales = [1.0, 0.5, 0.2, 0.1]
                pts_per_scale = n_local // len(local_scales)
                for scale in local_scales:
                    local_pts = best_x + rng.normal(0, scale, size=(pts_per_scale, dim))
                    local_pts = np.clip(local_pts, -5, 5)
                    candidates.extend(local_pts)
            
            candidates = np.array(candidates)
            
            # Predict mean and variance
            mean, var = predict_gp(candidates)
            
            # Compute EI
            # Note: GP predicts the normalized y? No, we used raw y in X_obs/y_obs.
            # So mean/var are in raw y space.
            ei = expected_improvement(mean, var, best_f)
            
            # Select best candidate
            best_idx = np.argmax(ei)
            x_next = candidates[best_idx]
            
            # Evaluate
            f_next = func(x_next)
            calls_made += 1
            
            if f_next < best_f:
                best_f = f_next
                best_x = x_next.copy()
            
            # Update GP
            X_obs = np.vstack([X_obs, x_next.reshape(1, -1)])
            y_obs = np.append(y_obs, f_next)
            
            # Update sigma_f occasionally to adapt to new scales
            if calls_made % 10 == 0:
                y_arr = y_obs
                var_y = np.var(y_arr)
                sigma_f = max(1.0, np.sqrt(var_y) * 2)
                
            # Safety check
            if time.time() > deadline:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
