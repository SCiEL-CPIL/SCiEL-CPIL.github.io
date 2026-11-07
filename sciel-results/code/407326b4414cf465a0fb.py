"""Model-based optimizer for BBOB f19 (Griewank-Rosenbrock).

Uses a Gaussian Process surrogate with multiple acquisition functions,
combined with L-BFGS-B local refinement. The GP model captures the
smooth structure of the function, while local optimization exploits
the smoothness for fast convergence.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """GP-based Bayesian optimizer with local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _gp_predict(self, X_new, X, y, length_scale=1.0, noise=1e-4):
        """Predict mean and variance at X_new using GP with RBF kernel."""
        n = X.shape[0]
        d = self.dim
        
        # Compute kernel matrix
        diff = X[:, None, :] - X[None, :, :]
        sq_dist = np.sum(diff ** 2, axis=-1)
        K = np.exp(-0.5 * sq_dist / (length_scale ** 2))
        K += noise * np.eye(n)
        
        # Solve for alpha
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        except np.linalg.LinAlgError:
            K += 1e-6 * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            except np.linalg.LinAlgError:
                # Fallback to simple distance-based prediction
                dists = np.sqrt(np.sum((X_new[:, None, :] - X[None, :, :]) ** 2, axis=-1))
                weights = np.exp(-dists / length_scale)
                weights /= weights.sum(axis=1, keepdims=True)
                mu = weights @ y
                sigma2 = np.var(y) * np.exp(-dists.min(axis=1) / length_scale)
                return mu, sigma2
        
        # Predict at new points
        K_s = np.exp(-0.5 * np.sum((X_new[:, None, :] - X[None, :, :]) ** 2, axis=-1) / (length_scale ** 2))
        mu = K_s @ alpha
        v = np.linalg.solve(L, K_s.T)
        sigma2 = np.var(y) - np.sum(v ** 2, axis=0)
        sigma2 = np.maximum(sigma2, 1e-10)
        
        return mu, sigma2

    def _acquisition_ei(self, mu, sigma, best_f, xi=0.0):
        """Expected Improvement acquisition function."""
        sigma = np.maximum(sigma, 1e-10)
        imp = best_f - mu - xi
        z = imp / sigma
        ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    def _acquisition_pi(self, mu, sigma, best_f, xi=0.0):
        """Probability of Improvement."""
        z = (best_f - mu - xi) / np.maximum(sigma, 1e-10)
        return norm.cdf(z)

    def _acquisition_ucb(self, mu, sigma, beta=2.0):
        """Upper Confidence Bound."""
        return mu + beta * sigma

    def _local_refine(self, x0, f0, deadline, max_iter=50):
        """L-BFGS-B local refinement."""
        def wrapper(x):
            x = np.clip(x, -5.0, 5.0)
            return float(self._f(x))
        
        try:
            res = minimize(wrapper, x0, method='L-BFGS-B',
                          bounds=[(-5.0, 5.0)] * self.dim,
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success or res.fun < f0:
                x_opt = np.clip(res.x, -5.0, 5.0)
                f_opt = float(res.fun)
                return x_opt, f_opt
        except Exception:
            pass
        
        # Fallback: Nelder-Mead
        try:
            res = minimize(wrapper, x0, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6,
                                   'adaptive': True})
            if res.fun < f0:
                x_opt = np.clip(res.x, -5.0, 5.0)
                f_opt = float(res.fun)
                return x_opt, f_opt
        except Exception:
            pass
        
        return x0, f0

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        self._f = func
        
        n = self.dim
        budget = self.budget
        
        # Initialize with a diverse set of points
        X_obs = []
        y_obs = []
        
        # Start with center and some random points
        x0 = np.zeros(n)
        f0 = float(func(x0))
        X_obs.append(x0)
        y_obs.append(f0)
        
        # Add a few random points for initial exploration
        for _ in range(min(10, budget - 1)):
            if time.time() > deadline - 30:
                break
            x = rng.uniform(-5.0, 5.0, size=n)
            f = float(func(x))
            X_obs.append(x)
            y_obs.append(f)
        
        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)
        
        best_x = X_obs[np.argmin(y_obs)].copy()
        best_f = float(np.min(y_obs))
        
        calls = len(X_obs)
        
        # Adaptive length scale
        length_scales = [0.5, 1.0, 2.0, 4.0]
        current_ls_idx = 0
        
        # Main optimization loop
        while calls < budget and time.time() < deadline - 20:
            # Update GP model with current observations
            mu, sigma2 = self._gp_predict(X_obs, X_obs, y_obs, 
                                          length_scale=length_scales[current_ls_idx])
            sigma = np.sqrt(sigma2)
            
            # Generate candidate points
            n_candidates = 100
            candidates = []
            
            # Mix of random and structured candidates
            # Random points
            rand_candidates = rng.uniform(-5.0, 5.0, size=(n_candidates // 2, n))
            candidates.append(rand_candidates)
            
            # Points near current best
            near_best = best_x + rng.normal(0, 0.5, size=(n_candidates // 4, n))
            near_best = np.clip(near_best, -5.0, 5.0)
            candidates.append(near_best)
            
            # Points along principal directions (rough PCA)
            if len(X_obs) >= n + 1:
                X_centered = X_obs - X_obs.mean(axis=0)
                try:
                    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
                    top_k = min(3, Vt.shape[0])
                    pca_candidates = []
                    for k in range(top_k):
                        direction = Vt[k]
                        offsets = rng.uniform(-2, 2, size=n_candidates // (4 * top_k))
                        for off in offsets:
                            pt = best_x + off * direction
                            pt = np.clip(pt, -5.0, 5.0)
                            pca_candidates.append(pt)
                    if pca_candidates:
                        candidates.append(np.array(pca_candidates))
                except Exception:
                    pass
            
            candidates = np.vstack(candidates)
            candidates = np.clip(candidates, -5.0, 5.0)
            
            # Predict at candidates
            mu_cand, sigma2_cand = self._gp_predict(candidates, X_obs, y_obs,
                                                     length_scale=length_scales[current_ls_idx])
            sigma_cand = np.sqrt(sigma2_cand)
            
            # Compute acquisition values
            ei = self._acquisition_ei(mu_cand, sigma_cand, best_f, xi=0.01)
            pi = self._acquisition_pi(mu_cand, sigma_cand, best_f, xi=0.01)
            ucb = self._acquisition_ucb(mu_cand, sigma_cand, beta=2.0)
            
            # Combine acquisition functions
            # Normalize each
            ei_norm = (ei - ei.min()) / (ei.max() - ei.min() + 1e-10)
            pi_norm = (pi - pi.min()) / (pi.max() - pi.min() + 1e-10)
            ucb_norm = (ucb - ucb.min()) / (ucb.max() - ucb.min() + 1e-10)
            
            # Weighted combination (adjust weights over time)
            progress = calls / budget
            w_ei = 0.5 + 0.3 * progress
            w_pi = 0.3 - 0.1 * progress
            w_ucb = 0.2 - 0.1 * progress
            
            acq = w_ei * ei_norm + w_pi * pi_norm + w_ucb * ucb_norm
            
            # Select top candidates
            top_k = min(5, len(acq))
            top_indices = np.argsort(-acq)[:top_k]
            
            # Evaluate top candidates
            evaluated = False
            for idx in top_indices:
                if calls >= budget or time.time() > deadline - 10:
                    break
                x_new = candidates[idx]
                f_new = float(func(x_new))
                calls += 1
                
                X_obs = np.vstack([X_obs, x_new])
                y_obs = np.append(y_obs, f_new)
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                    evaluated = True
                    # Try local refinement
                    if time.time() < deadline - 30 and calls < budget - 10:
                        x_ref, f_ref = self._local_refine(x_new, f_new, deadline, max_iter=20)
                        if f_ref < best_f:
                            best_f = f_ref
                            best_x = x_ref.copy()
                            # Add refined point to observations
                            X_obs = np.vstack([X_obs, x_ref])
                            y_obs = np.append(y_obs, f_ref)
                            calls += 1  # Approximate additional calls
            
            # Adapt length scale based on progress
            if calls % 20 == 0:
                # Try different length scales
                if current_ls_idx < len(length_scales) - 1:
                    current_ls_idx += 1
                else:
                    current_ls_idx = 0
        
        # Final local refinement if time permits
        if time.time() < deadline - 15 and calls < budget - 5:
            x_final, f_final = self._local_refine(best_x, best_f, deadline, max_iter=30)
            if f_final < best_f:
                best_f = f_final
                best_x = x_final.copy()
        
        return best_x, best_f
