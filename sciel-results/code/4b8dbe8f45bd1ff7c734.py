"""Surrogate-assisted Bayesian Optimizer for BBOB f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """GP-based Bayesian Optimizer with local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        X_obs = []
        y_obs = []

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            X_obs.append(x.copy())
            y_obs.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial Sampling (Stratified/LHS-like)
        n_init = 20
        # Use a simple stratified sampling to ensure coverage
        # Divide each dimension into bins? Or just random with rejection?
        # For simplicity and speed, random uniform is okay, but LHS is better.
        # Let's use a simple grid + random mix.
        init_points = []
        # Center
        init_points.append(np.zeros(dim))
        # Axis points
        for d in range(dim):
            for val in [-3.0, 3.0]:
                p = np.zeros(dim)
                p[d] = val
                init_points.append(p)
        
        # Fill with random
        while len(init_points) < n_init:
            init_points.append(rng.uniform(-4.5, 4.5, dim))
            
        init_points = np.array(init_points[:n_init])
        
        for i in range(n_init):
            if calls_made >= budget or time.time() > deadline:
                break
            evaluate(init_points[i])

        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # GP Parameters
        # Matérn 5/2
        # Fixed hyperparameters to avoid expensive optimization
        L = 1.5  # Length scale
        sigma_f = 1.0
        sigma_n = 1e-6

        def matern52(R):
            # R is distance matrix
            return (1 + np.sqrt(5)*R + (5/3)*R**2) * np.exp(-np.sqrt(5)*R)

        def fit_gp(X, y):
            # Standardize y
            y_mean = np.mean(y)
            y_std = np.std(y)
            if y_std < 1e-10:
                y_std = 1.0
            y_std = (y - y_mean) / y_std
            
            n = len(X)
            # Compute distance matrix
            # X is (n, dim)
            # dist[i,j] = ||X[i] - X[j]||
            # Using broadcasting
            X_norm = np.sum(X**2, axis=1)[:, np.newaxis]
            dist_sq = X_norm + X_norm.T - 2 * X @ X.T
            np.maximum(dist_sq, 0, out=dist_sq)
            R = np.sqrt(dist_sq) / L
            
            K = matern52(R) * sigma_f**2
            K[np.diag_indices_from(K)] += sigma_n**2
            
            # Solve K alpha = y_std
            try:
                alpha = np.linalg.solve(K, y_std)
            except np.linalg.LinAlgError:
                # Fallback to least squares if singular
                alpha = np.linalg.lstsq(K, y_std, rcond=None)[0]
                
            return K, alpha, y_mean, y_std

        def predict_gp(x_new, K, alpha, y_mean, y_std, X_obs_arr):
            # x_new is (dim,) or (n_new, dim)
            single = False
            if x_new.ndim == 1:
                x_new = x_new.reshape(1, -1)
                single = True
                
            n_new = x_new.shape[0]
            # Compute distances between x_new and X_obs
            X_norm = np.sum(X_obs_arr**2, axis=1)[:, np.newaxis]
            xn_norm = np.sum(x_new**2, axis=1)[:, np.newaxis]
            dist_sq = xn_norm + X_norm.T - 2 * x_new @ X_obs_arr.T
            np.maximum(dist_sq, 0, out=dist_sq)
            R = np.sqrt(dist_sq) / L
            
            K_x = matern52(R) * sigma_f**2 # (n_new, n_obs)
            
            mu_std = K_x @ alpha # (n_new,)
            mu = mu_std * y_std + y_mean
            
            # Variance
            k_xx = sigma_f**2 # Scalar for each point
            var_std = k_xx - np.sum(K_x**2, axis=1)
            var = var_std * y_std**2
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)
            
            if single:
                return mu[0], std[0]
            return mu, std

        def acquisition_ei(x, f_best_std):
            # x is (dim,)
            mu, std = predict_gp(x, K_gp, alpha_gp, y_mean_gp, y_std_gp, X_obs_arr)
            if std < 1e-10:
                return 0.0
            Z = (mu - f_best_std) / std
            ei = (mu - f_best_std) * norm.cdf(Z) + std * norm.pdf(Z)
            return ei

        # Phase 2: GP BO Loop
        # Reserve some budget for polish
        reserve = 30
        max_gp_calls = budget - reserve
        
        while calls_made < max_gp_calls:
            if time.time() > deadline:
                break
                
            # Update GP
            X_obs_arr = np.array(X_obs)
            y_obs_arr = np.array(y_obs)
            
            # If we have too few points, just random sample
            if len(X_obs) < 5:
                x_new = rng.uniform(-4.5, 4.5, dim)
                evaluate(x_new)
                continue

            K_gp, alpha_gp, y_mean_gp, y_std_gp = fit_gp(X_obs_arr, y_obs_arr)
            f_best_std = (best_f - y_mean_gp) / y_std_gp

            # Optimize EI
            # Use multiple starts for Nelder-Mead
            best_ei = -np.inf
            best_x_ei = None
            
            # Starts:
            # 1. Current best
            starts = [best_x.copy()]
            # 2. Random points
            for _ in range(10):
                starts.append(rng.uniform(-4.5, 4.5, dim))
            # 3. Points near best
            for _ in range(5):
                starts.append(best_x + 0.5 * rng.standard_normal(dim))
                
            for s in starts:
                s = np.clip(s, -5.0, 5.0)
                # Minimize -EI
                res = minimize(
                    lambda x: -acquisition_ei(x, f_best_std),
                    s,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 0.01, 'fatol': 1e-4}
                )
                if res.success or res.fun < -best_ei:
                    ei_val = -res.fun
                    if ei_val > best_ei:
                        best_ei = ei_val
                        best_x_ei = res.x

            if best_x_ei is None:
                best_x_ei = rng.uniform(-4.5, 4.5, dim)
                
            evaluate(best_x_ei)

        # Phase 3: Local Polish
        if calls_made < budget and time.time() < deadline and best_x is not None:
            xp = best_x.copy()
            fp = best_f
            sp = 0.1
            # Local random walk
            while calls_made < budget and time.time() < deadline:
                y = xp + sp * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy < fp:
                    xp, fp = y, fy
                    sp *= 0.9
                else:
                    sp *= 0.8
                if sp < 0.01:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
