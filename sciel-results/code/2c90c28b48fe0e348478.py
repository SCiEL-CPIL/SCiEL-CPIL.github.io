import numpy as np
import time

class Optimizer:
    """Bayesian-optimization-style optimizer with GP surrogate and EI acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Safety margin: 330s limit, stop at 315s
        deadline = time.time() + 315
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        # Store observations
        X_obs = []
        y_obs = []
        
        # Initial design: Latin Hypercube for good coverage
        n_init = min(30, budget // 10)
        
        # Latin Hypercube Sampling
        lhd = np.zeros((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            lhd[:, d] = (perm + rng.uniform(size=n_init)) / n_init
        lhd = lhd * 10.0 - 5.0  # Map to [-5, 5]
        
        # Evaluate initial design
        for x in lhd:
            if calls >= budget or time.time() > deadline:
                break
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            if not np.isfinite(val):
                val = 1e10
            calls += 1
            X_obs.append(x_clipped)
            y_obs.append(val)
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
        
        if best_x is None:
            return np.zeros(dim), np.inf
        
        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)
        
        # Normalize y for GP stability
        y_mean = np.mean(y_obs)
        y_std = np.std(y_obs) + 1e-10
        y_norm = (y_obs - y_mean) / y_std
        
        def gp_predict(X_new, X_train, y_train):
            """Gaussian Process prediction with RBF kernel."""
            # RBF kernel
            length_scale = 1.0
            sigma_f = 1.0
            noise = 1e-6
            
            # Compute kernel matrix K
            K = np.zeros((len(X_train), len(X_train)))
            for i in range(len(X_train)):
                for j in range(len(X_train)):
                    diff = X_train[i] - X_train[j]
                    K[i, j] = sigma_f**2 * np.exp(-0.5 * np.sum(diff**2) / (length_scale**2))
            K += noise * np.eye(len(X_train))
            
            # Compute K_*
            K_star = np.zeros((len(X_new), len(X_train)))
            for i in range(len(X_new)):
                for j in range(len(X_train)):
                    diff = X_new[i] - X_train[j]
                    K_star[i, j] = sigma_f**2 * np.exp(-0.5 * np.sum(diff**2) / (length_scale**2))
            
            # Solve for mean and variance
            try:
                alpha = np.linalg.solve(K, y_train)
                mu = K_star @ alpha
                v = np.linalg.solve(K, K_star.T)
                var = sigma_f**2 - np.sum(K_star * v, axis=1)
                var = np.maximum(var, 1e-10)
                std = np.sqrt(var)
            except np.linalg.LinAlgError:
                mu = np.zeros(len(X_new))
                std = np.ones(len(X_new))
            
            return mu, std
        
        def ei_acquisition(X_cand, y_best):
            """Expected Improvement acquisition function."""
            mu, std = gp_predict(X_cand, X_obs, y_norm)
            # Convert y_best to normalized scale
            y_best_norm = (y_best - y_mean) / y_std
            # EI = (mu - y_best) * Phi((mu - y_best)/std) + std * phi((mu - y_best)/std)
            z = (mu - y_best_norm) / (std + 1e-10)
            from scipy.stats import norm
            phi = norm.pdf(z)
            Phi = norm.cdf(z)
            ei = (mu - y_best_norm) * Phi + std * phi
            return ei
        
        # Main BO loop
        while calls < budget and time.time() < deadline:
            # Generate candidate points
            n_cand = min(100, (budget - calls) * 2)
            X_cand = rng.uniform(-5, 5, (n_cand, dim))
            
            # Also add points near current best
            n_near = min(50, (budget - calls))
            X_near = best_x + rng.normal(0, 0.5, (n_near, dim))
            X_near = np.clip(X_near, -5, 5)
            X_cand = np.vstack([X_cand, X_near])
            
            # Compute EI
            ei_vals = ei_acquisition(X_cand, best_f)
            
            # Select top candidates
            n_select = min(5, len(ei_vals), budget - calls)
            top_idx = np.argsort(ei_vals)[-n_select:]
            
            # Evaluate selected points
            for idx in top_idx:
                if calls >= budget or time.time() > deadline:
                    break
                x_new = X_cand[idx]
                val = func(x_new)
                if not np.isfinite(val):
                    val = 1e10
                calls += 1
                X_obs = np.vstack([X_obs, x_new])
                y_obs = np.append(y_obs, val)
                y_norm = np.append(y_norm, (val - y_mean) / y_std)
                if val < best_f:
                    best_f = val
                    best_x = x_new.copy()
                    # Update y_mean and y_std
                    y_mean = np.mean(y_obs)
                    y_std = np.std(y_obs) + 1e-10
                    y_norm = (y_obs - y_mean) / y_std
            
            # If we've used most of our budget, do a final local polish
            if calls > budget * 0.8:
                # Simple local search with shrinking step
                sigma = 0.3
                for _ in range(min(30, budget - calls)):
                    if calls >= budget or time.time() > deadline:
                        break
                    x_try = best_x + rng.normal(0, sigma, dim)
                    x_try = np.clip(x_try, -5, 5)
                    val = func(x_try)
                    if not np.isfinite(val):
                        val = 1e10
                    calls += 1
                    if val < best_f:
                        best_f = val
                        best_x = x_try.copy()
                    else:
                        sigma *= 0.9
    
        return best_x, best_f
