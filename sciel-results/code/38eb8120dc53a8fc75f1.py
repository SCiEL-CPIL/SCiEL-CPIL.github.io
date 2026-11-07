"""Bayesian Optimization with RBF surrogate for BBOB f17."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from math import log, exp, sqrt


class Optimizer:
    """Bayesian Optimization with RBF kernel and EI acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # Phase 1: Initial exploration with space-filling points
        n_init = min(10, budget // 5)
        
        # Use a mix of center and boundary points plus random
        X = []
        # Center
        X.append(np.zeros(n))
        # Random points
        for _ in range(n_init - 1):
            X.append(rng.uniform(-4.5, 4.5, n))
        X = np.array(X)
        
        Y = np.array([func(x) for x in X])
        best_idx = np.argmin(Y)
        best_x = X[best_idx].copy()
        best_y = Y[best_idx]
        
        calls_made = len(X)
        
        # RBF kernel parameters
        # Adaptive length scale based on data spread
        def rbf_kernel(X1, X2, length_scale):
            D2 = cdist(X1, X2) ** 2
            return np.exp(-D2 / (2 * length_scale ** 2))
        
        def compute_length_scale(X_obs):
            """Compute a reasonable length scale from observed data."""
            if len(X_obs) < 2:
                return 1.0
            # Use median distance
            D = cdist(X_obs, X_obs)
            mask = ~np.eye(len(X_obs), dtype=bool)
            if not mask.any():
                return 1.0
            med_dist = np.median(D[mask])
            return max(0.5, med_dist * 0.5)
        
        def fit_rbf(X_obs, Y_obs, length_scale):
            """Fit RBF surrogate, return mean and std predictions."""
            m = len(X_obs)
            K = rbf_kernel(X_obs, X_obs, length_scale)
            # Add small jitter for numerical stability
            K += 1e-8 * np.eye(m)
            
            # Normalize Y for stability
            y_mean = np.mean(Y_obs)
            y_std = max(np.std(Y_obs), 1e-8)
            Y_norm = (Y_obs - y_mean) / y_std
            
            try:
                alpha = np.linalg.solve(K, Y_norm)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, Y_norm, rcond=None)[0]
            
            return y_mean, y_std, alpha, K
        
        def predict(X_new, X_obs, Y_obs, length_scale, y_mean, y_std, alpha):
            """Predict mean and std at new points."""
            K_new = rbf_kernel(X_new, X_obs, length_scale)
            K_obs = rbf_kernel(X_obs, X_obs, length_scale) + 1e-8 * np.eye(len(X_obs))
            
            # Mean prediction
            mu = K_new @ alpha * y_std + y_mean
            
            # Variance prediction (simplified, using prior variance)
            v = 1.0 + 1e-8  # prior variance
            try:
                K_inv = np.linalg.inv(K_obs)
                v_pred = v - K_new @ K_inv @ K_new.T
            except np.linalg.LinAlgError:
                v_pred = np.full(len(X_new), v)
            
            std = np.sqrt(np.maximum(v_pred, 1e-10))
            return mu, std
        
        def ei_acquisition(mu, std, best_y_obs, xi=0.01):
            """Expected Improvement acquisition function."""
            improvement = best_y_obs - mu - xi
            # Avoid division by zero
            std_safe = np.maximum(std, 1e-10)
            z = improvement / std_safe
            ei = improvement * (z * 0.3989 + 1) + std_safe * 0.2499  # approx normal CDF/PDF
            # More accurate: use erfc for tail
            from math import erf
            ei = np.zeros_like(mu)
            pos = improvement > 0
            ei[pos] = improvement[pos] * (1 - 0.5 * (1 - erf(z[pos] / sqrt(2)))) + std_safe[pos] * 0.3989 * exp(-0.5 * z[pos]**2)
            return ei
        
        # Phase 2: Bayesian Optimization loop
        polish_counter = 0
        no_improve_count = 0
        polish_threshold = 50  # polish every N iterations
        
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Update length scale
            length_scale = compute_length_scale(X)
            y_mean, y_std, alpha, K_obs = fit_rbf(X, Y, length_scale)
            
            # Generate candidate points for acquisition
            n_cand = 2000
            X_cand = rng.uniform(-4.9, 4.9, (n_cand, n))
            
            # Also add points near the current best
            n_local = 500
            X_local = best_x + rng.uniform(-1.0, 1.0, (n_local, n))
            X_local = np.clip(X_local, -4.9, 4.9)
            X_cand = np.vstack([X_cand, X_local])
            
            # Predict
            mu, std = predict(X_cand, X, Y, length_scale, y_mean, y_std, alpha)
            
            # EI acquisition
            ei = ei_acquisition(mu, std, best_y)
            
            # Select best candidate
            best_cand_idx = np.argmax(ei)
            x_new = X_cand[best_cand_idx]
            
            # Evaluate
            y_new = func(x_new)
            calls_made += 1
            
            # Update observations
            X = np.vstack([X, x_new.reshape(1, -1)])
            Y = np.append(Y, y_new)
            
            if y_new < best_y:
                best_y = y_new
                best_x = x_new.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Periodic local polish: if stuck, do a small local search
            if no_improve_count >= polish_threshold:
                no_improve_count = 0
                polish_counter += 1
                # Do a few local perturbations
                n_polish = min(10, budget - calls_made)
                for _ in range(n_polish):
                    if time.time() > deadline or calls_made >= budget:
                        break
                    x_p = best_x + rng.uniform(-0.3, 0.3, n)
                    x_p = np.clip(x_p, -4.9, 4.9)
                    y_p = func(x_p)
                    calls_made += 1
                    X = np.vstack([X, x_p.reshape(1, -1)])
                    Y = np.append(Y, y_p)
                    if y_p < best_y:
                        best_y = y_p
                        best_x = x_p.copy()
        
        return best_x, best_y
# EVOLVE-BLOCK-END
