"""Gaussian Process Surrogate-based Bayesian Optimizer for Rosenbrock."""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """GP-based Bayesian Optimizer for continuous black-box functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Data structures for GP
        X = []
        Y = []

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline - 5:
                raise ValueError("Time exceeded")
            calls_used += 1
            f_val = float(func(x))
            
            # Update best
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
            return f_val

        # Phase 1: Initial Exploration (Latin Hypercube + Origin)
        # Use a small LHS to cover the space efficiently
        n_init = min(15, budget // 10)
        if n_init < 5:
            n_init = 5
            
        # Add origin
        init_points = [np.zeros(dim)]
        
        # Generate LHS points
        # Simple LHS: divide each dimension into n_init bins, shuffle
        lhs = np.zeros((n_init, dim))
        for d in range(dim):
            bins = np.arange(n_init) / n_init
            rng.shuffle(bins)
            # Map to [-5, 5]
            lhs[:, d] = bins * 10 - 5
            
        init_points.extend([lhs[i] for i in range(n_init)])
        
        # Evaluate initial points
        for x in init_points:
            try:
                f_val = safe_call(x)
                X.append(x)
                Y.append(f_val)
            except Exception:
                break

        if best_x is None:
            # Fallback if even first call fails or budget is tiny
            return np.zeros(dim), 0.0

        # Phase 2: GP Surrogate Loop
        # We will fit a simple GP manually using RBF kernel to avoid heavy scipy.stats dependencies
        # or use a lightweight implementation.
        # Given 500 budget, GP fitting is O(N^3) per iteration. N starts at ~16, grows to 500.
        # 500^3 is 125M operations, which might be too slow in Python if done naively in a loop.
        # However, we can limit the GP training set size or use a sparse GP approximation.
        # Alternatively, use a simpler surrogate like Kriging with a limited history or restart GP.
        
        # Let's implement a simple RBF GP with a cap on the number of points used for fitting
        # to keep it fast. If N > 100, we use the most recent 100 points + the best point.
        
        def compute_gp(X_train, Y_train, x_new, length_scale=1.0, sigma_f=1.0, noise=1e-6):
            """
            Compute mean and variance of GP at x_new.
            X_train: (N, D)
            Y_train: (N,)
            x_new: (D,)
            """
            N = len(X_train)
            if N == 0:
                return 0.0, 1.0
            
            # Kernel matrix K
            # K_ij = sigma_f^2 * exp(-0.5 * ||x_i - x_j||^2 / l^2)
            # For efficiency, compute distances
            diff = X_train[:, np.newaxis, :] - x_new[np.newaxis, np.newaxis, :]
            sq_dist = np.sum(diff**2, axis=2)
            K = sigma_f**2 * np.exp(-0.5 * sq_dist / length_scale**2)
            
            # K + noise*I for training
            K_train = sigma_f**2 * np.exp(-0.5 * np.sum((X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :])**2, axis=2) / length_scale**2)
            K_train[np.diag_indices_from(K_train)] += noise
            
            try:
                # Solve linear system
                alpha = np.linalg.solve(K_train, Y_train)
                mean = np.dot(K, alpha)
                
                # Variance
                v = np.linalg.solve(K_train, K)
                var = sigma_f**2 - np.dot(K, v)
                var = max(var, 1e-8)
            except np.linalg.LinAlgError:
                # Fallback if singular
                mean = np.mean(Y_train)
                var = np.var(Y_train) + 1.0
                
            return mean, var

        def expected_improvement(x, current_best, length_scale=1.0, sigma_f=1.0, noise=1e-6):
            """
            Compute EI for a candidate point x.
            """
            mean, var = compute_gp(X_active, Y_active, x, length_scale, sigma_f, noise)
            std = np.sqrt(var)
            if std < 1e-8:
                return 0.0
            z = (current_best - mean) / std
            # Normal CDF and PDF
            from scipy.stats import norm
            ei = (current_best - mean) * norm.cdf(z) + std * norm.pdf(z)
            return ei

        # Adaptive length scale and sigma_f based on data
        # Initial heuristic
        length_scale = 1.5
        sigma_f = 1.0
        
        # Main BO loop
        while calls_used < budget and time.time() < deadline - 10:
            # Update GP parameters based on current data
            Y_arr = np.array(Y)
            Y_min = np.min(Y_arr)
            Y_max = np.max(Y_arr)
            Y_range = Y_max - Y_min
            if Y_range < 1e-6:
                Y_range = 1.0
            sigma_f = Y_range
            
            # Select active training set to keep GP fast
            # Use all points if N < 50, else subsample
            N = len(X)
            if N > 50:
                # Keep best point and most recent 49 points? Or random subsample?
                # Random subsample is better for coverage
                indices = rng.choice(N, size=50, replace=False)
                X_active = np.array(X)[indices]
                Y_active = np.array(Y)[indices]
            else:
                X_active = np.array(X)
                Y_active = np.array(Y)
                
            current_best = best_f
            
            # Optimize EI over the domain using L-BFGS-B
            # We need to minimize -EI
            def neg_ei(x):
                try:
                    # Clip to bounds for safety
                    x_clipped = np.clip(x, -5, 5)
                    ei = expected_improvement(x_clipped, current_best, length_scale, sigma_f)
                    return -ei
                except Exception:
                    return 1e6

            # Multi-start for EI optimization
            n_starts = 5
            best_ei_x = None
            best_ei_val = -np.inf
            
            for i in range(n_starts):
                # Start from random point or best point
                if i == 0:
                    start_x = best_x.copy()
                else:
                    start_x = rng.uniform(-5, 5, dim)
                    
                try:
                    res = minimize(
                        neg_ei, 
                        start_x, 
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50, 'gtol': 1e-4}
                    )
                    if res.fun < best_ei_val:
                        best_ei_val = res.fun
                        best_ei_x = res.x
                except Exception:
                    continue
            
            if best_ei_x is None:
                # Fallback to random
                best_ei_x = rng.uniform(-5, 5, dim)
                
            # Evaluate the selected point
            x_new = np.clip(best_ei_x, -5, 5)
            try:
                f_val = safe_call(x_new)
                X.append(x_new)
                Y.append(f_val)
            except Exception:
                break
                
            # Periodically adapt length scale based on distance between best points?
            # Simple heuristic: if we haven't improved in 10 steps, increase exploration (larger length scale)
            # For now, keep it fixed or simple adaptive
            
        return best_x, best_f
