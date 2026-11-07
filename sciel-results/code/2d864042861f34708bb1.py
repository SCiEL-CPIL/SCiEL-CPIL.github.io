"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid RBF surrogate + L-BFGS-B optimizer for f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        d = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline

        calls = 0
        best_x = np.zeros(d)
        best_fx = np.inf
        
        def safe_func(x):
            nonlocal calls, best_x, best_fx
            if calls >= budget:
                return best_fx
            x = np.clip(x, -5, 5)
            calls += 1
            fx = float(func(x))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global exploration with RBF surrogate model
        # Collect initial samples
        rng = np.random.RandomState(42)
        
        # Generate diverse initial points
        n_initial = 20
        X_samples = []
        f_samples = []
        
        # Center
        x0 = np.zeros(d)
        fx0 = safe_func(x0)
        X_samples.append(x0.copy())
        f_samples.append(fx0)
        
        # Random points
        for _ in range(n_initial - 1):
            x = rng.uniform(-5, 5, size=d)
            fx = safe_func(x)
            X_samples.append(x.copy())
            f_samples.append(fx)
        
        X_samples = np.array(X_samples)
        f_samples = np.array(f_samples)
        
        # RBF surrogate model
        def rbf_kernel(X1, X2, length_scale):
            """Gaussian RBF kernel."""
            D = np.linalg.norm(X1[:, np.newaxis, :] - X2[np.newaxis, :, :], axis=2)
            return np.exp(-D**2 / (2 * length_scale**2))
        
        def train_rbf(X, y, length_scale=1.0):
            """Train RBF model."""
            K = rbf_kernel(X, X, length_scale)
            # Add small regularization for numerical stability
            K += 1e-8 * np.eye(len(X))
            try:
                theta = np.linalg.solve(K, y)
            except np.linalg.LinAlgError:
                theta = np.linalg.lstsq(K, y, rcond=None)[0]
            return theta, K
        
        def predict_rbf(X_new, X_train, theta, length_scale=1.0):
            """Predict with RBF model."""
            K = rbf_kernel(X_new, X_train, length_scale)
            return K @ theta
        
        def acquisition_function(X_cand, X_train, theta, best_y, length_scale=1.0, xi=0.01):
            """Expected Improvement acquisition."""
            mu = predict_rbf(X_cand, X_train, theta, length_scale)
            # Approximate uncertainty using kernel values
            K_train = rbf_kernel(X_train, X_train, length_scale)
            K_train += 1e-8 * np.eye(len(X_train))
            try:
                K_inv = np.linalg.inv(K_train)
            except np.linalg.LinAlgError:
                K_inv = np.linalg.pinv(K_train)
            
            K_cand = rbf_kernel(X_cand, X_train, length_scale)
            var = 1.0 - K_cand @ K_inv @ K_cand.T
            var = np.maximum(var, 1e-8)
            sigma = np.sqrt(var)
            
            # Expected improvement
            improvement = best_y - mu - xi
            z = improvement / sigma
            from scipy.stats import norm
            EI = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            return EI
        
        # Phase 1: Active learning with RBF
        rbf_length = 1.5
        theta, _ = train_rbf(X_samples, f_samples, rbf_length)
        
        n_rbf_steps = min(30, budget - calls - 50)  # Reserve budget for local search
        
        for step in range(n_rbf_steps):
            if not time_ok() or calls >= budget - 50:
                break
            
            # Generate candidate points
            n_cand = 100
            X_cand = rng.uniform(-5, 5, size=(n_cand, d))
            
            # Evaluate acquisition
            EI = acquisition_function(X_cand, X_samples, theta, best_fx, rbf_length)
            
            # Select best candidate
            idx = np.argmax(EI)
            x_next = X_cand[idx]
            
            # Evaluate
            fx_next = safe_func(x_next)
            
            # Update samples
            X_samples = np.vstack([X_samples, x_next])
            f_samples = np.append(f_samples, fx_next)
            
            # Retrain RBF
            theta, _ = train_rbf(X_samples, f_samples, rbf_length)
        
        # Phase 2: Local refinement with L-BFGS-B from multiple starting points
        bounds = [(-5, 5)] * d
        
        # Use best RBF prediction point and diverse points
        start_points = [best_x.copy()]
        
        # Add points from RBF exploration
        top_k = min(5, len(X_samples))
        if top_k > 1:
            # Get top-k diverse points from samples
            indices = np.argsort(f_samples)[:top_k]
            for i in indices:
                start_points.append(X_samples[i].copy())
        
        # Add some random points
        for _ in range(3):
            start_points.append(rng.uniform(-5, 5, size=d))
        
        # Run L-BFGS-B from each starting point
        for x0 in start_points:
            if not time_ok() or calls >= budget - 5:
                break
            
            def local_func(v):
                nonlocal calls
                if calls >= budget:
                    return best_fx
                return safe_func(v)
            
            try:
                res = minimize(
                    local_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 30, 'maxls': 20, 'ftol': 1e-14, 'gtol': 1e-10, 'eps': 1e-8}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
