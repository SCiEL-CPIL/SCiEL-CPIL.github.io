# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.spatial.distance import cdist


class Optimizer:
    """Bayesian Optimization with Gaussian Process surrogate and Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _gp_predict(self, X_new, X_train, y_train, length_scale, noise_var):
        """Predict mean and variance using a simple GP with RBF kernel."""
        # Compute kernel matrix K(x, x')
        dists = cdist(X_new, X_train)
        K = np.exp(-0.5 * (dists ** 2) / (length_scale ** 2))
        
        # Train kernel matrix
        K_train = np.exp(-0.5 * (cdist(X_train, X_train) ** 2) / (length_scale ** 2))
        K_train += noise_var * np.eye(len(X_train))
        
        # Solve for weights
        try:
            L = np.linalg.cholesky(K_train)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        except np.linalg.LinAlgError:
            K_train += 1e-8 * np.eye(len(X_train))
            try:
                L = np.linalg.cholesky(K_train)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            except np.linalg.LinAlgError:
                alpha = np.linalg.pinv(K_train) @ y_train
        
        # Predictions
        mu = K @ alpha
        v = np.linalg.solve(L, K.T)
        var = 1.0 - np.sum(v ** 2, axis=1)
        var = np.maximum(var, 1e-12)
        
        return mu, var

    def _expected_improvement(self, mu, var, best_f, xi=0.01):
        """Compute Expected Improvement acquisition function."""
        sigma = np.sqrt(var)
        improved = best_f - mu - xi
        z = improved / (sigma + 1e-12)
        ei = improved * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        rng = np.random.default_rng(42)

        # Initialize with a few diverse points
        X_obs = []
        y_obs = []
        best_f = np.inf
        best_x = None

        # Initial sampling: center + corners + random
        init_points = [np.zeros(dim)]
        for d in range(dim):
            for val in [-3.0, 3.0]:
                p = np.zeros(dim)
                p[d] = val
                init_points.append(p)
        
        # Add some random points
        for _ in range(5):
            init_points.append(rng.uniform(-4, 4, dim))
        
        # Evaluate initial points
        for p in init_points:
            p = np.clip(p, -5, 5)
            f = func(p)
            X_obs.append(p)
            y_obs.append(f)
            if f < best_f:
                best_f = f
                best_x = p.copy()
        
        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)
        calls = len(X_obs)

        # GP hyperparameters (tuned for 5D space)
        length_scale = 1.5
        noise_var = 1e-6

        # Bayesian optimization loop
        while calls < budget:
            if time.time() > deadline:
                break

            # Generate candidate points: mix of local and global
            n_candidates = 2000
            
            # Local candidates around best point
            if best_x is not None:
                local_radius = 1.0
                local_candidates = best_x + rng.standard_normal((n_candidates // 2, dim)) * local_radius
            else:
                local_candidates = rng.uniform(-5, 5, (n_candidates // 2, dim))

            # Global candidates
            global_candidates = rng.uniform(-5, 5, (n_candidates - n_candidates // 2, dim))
            
            candidates = np.vstack([local_candidates, global_candidates])
            candidates = np.clip(candidates, -5, 5)

            # Predict with GP
            mu, var = self._gp_predict(candidates, X_obs, y_obs, length_scale, noise_var)
            
            # Compute EI
            ei = self._expected_improvement(mu, var, best_f, xi=0.01)
            
            # Select best candidate
            idx = np.argmax(ei)
            x_next = candidates[idx]
            f_next = func(x_next)
            calls += 1

            # Update observations
            X_obs = np.vstack([X_obs, x_next.reshape(1, -1)])
            y_obs = np.append(y_obs, f_next)

            if f_next < best_f:
                best_f = f_next
                best_x = x_next.copy()
                # Adaptive: reduce noise as we converge
                noise_var = max(1e-8, noise_var * 0.95)

            # Adapt length scale based on spread of observations
            if len(y_obs) > 10:
                # Estimate characteristic length from data
                dists = cdist(X_obs[-10:], X_obs[:-10])
                length_scale = max(0.5, np.median(dists) * 0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
