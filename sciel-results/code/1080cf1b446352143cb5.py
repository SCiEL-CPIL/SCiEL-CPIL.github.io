"""Surrogate-assisted optimizer for BBOB f12 (Bent Cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-fidelity surrogate-assisted search using RBF + L-BFGS-B."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _rbf_predict(self, X_new, X_obs, y_obs, eps=1.0):
        """
        Predict using a Gaussian RBF interpolant.
        X_obs: (n, d), y_obs: (n,)
        X_new: (m, d)
        """
        n = len(X_obs)
        d = self.dim
        # Compute distance matrix between X_new and X_obs
        # Use efficient computation
        diff = X_new[:, np.newaxis, :] - X_obs[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        K = np.exp(-eps * dist_sq)
        # Solve for weights: K_obs @ w = y_obs
        # K_obs is n x n
        diff_obs = X_obs[:, np.newaxis, :] - X_obs[np.newaxis, :, :]
        dist_sq_obs = np.sum(diff_obs ** 2, axis=2)
        K_obs = np.exp(-eps * dist_sq_obs)
        # Add small regularization for numerical stability
        K_obs += 1e-8 * np.eye(n)
        try:
            w = np.linalg.solve(K_obs, y_obs)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(K_obs, y_obs, rcond=None)[0]
        return K @ w

    def _rbf_gradient(self, X_new, X_obs, y_obs, eps=1.0):
        """
        Compute gradient of RBF prediction w.r.t. X_new.
        """
        n = len(X_obs)
        d = self.dim
        diff = X_new[:, np.newaxis, :] - X_obs[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        K = np.exp(-eps * dist_sq)
        diff_obs = X_obs[:, np.newaxis, :] - X_obs[np.newaxis, :, :]
        dist_sq_obs = np.sum(diff_obs ** 2, axis=2)
        K_obs = np.exp(-eps * dist_sq_obs)
        K_obs += 1e-8 * np.eye(n)
        try:
            w = np.linalg.solve(K_obs, y_obs)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(K_obs, y_obs, rcond=None)[0]
        
        # Gradient: d/dx_new K(x_new, x_obs) = -2*eps*(x_new - x_obs)*exp(-eps*dist_sq)
        # grad[i, j] = sum_k K[i,k] * (-2*eps*(x_new[i,j] - x_obs[k,j])) * w[k]
        # = -2*eps * sum_k K[i,k] * (x_new[i,j] - x_obs[k,j]) * w[k]
        # = -2*eps * (x_new[i,j] * sum_k K[i,k]*w[k] - sum_k K[i,k]*x_obs[k,j]*w[k])
        
        Kw = K @ w  # (m,)
        Kxw = K @ (w[:, np.newaxis] * X_obs)  # (m, d)
        grad = -2 * eps * (X_new * Kw[:, np.newaxis] - Kxw)
        return grad

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Track time
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        # Storage for observed points
        X_obs = []
        y_obs = []
        
        # Initial exploration: sample diverse points
        # Start with center and some random points
        init_points = []
        init_points.append(np.zeros(dim))
        # Add points along each axis
        for i in range(dim):
            for sign in [1, -1]:
                p = np.zeros(dim)
                p[i] = sign * 3.0
                init_points.append(p)
        # Add some random points
        for _ in range(5):
            init_points.append(rng.uniform(-4, 4, dim))
        
        # Evaluate initial points
        for p in init_points:
            p = np.clip(p, -5, 5)
            fp = func(p)
            X_obs.append(p)
            y_obs.append(fp)
        
        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)
        
        best_x = X_obs[np.argmin(y_obs)]
        best_f = np.min(y_obs)
        
        # Adaptive epsilon for RBF
        eps = 0.5
        
        # Main optimization loop
        calls_made = len(X_obs)
        
        while calls_made < budget:
            # Check time
            if time.time() > deadline - 5:
                break
            
            # Find current best
            idx_best = np.argmin(y_obs)
            best_x = X_obs[idx_best]
            best_f = y_obs[idx_best]
            
            # Use L-BFGS-B on the surrogate to find a new candidate
            # Start from current best
            x0 = best_x.copy()
            
            def surrogate(x):
                x = np.clip(x, -5, 5)
                return self._rbf_predict(x[np.newaxis, :], X_obs, y_obs, eps)[0]
            
            def surrogate_grad(x):
                x = np.clip(x, -5, 5)
                return self._rbf_gradient(x[np.newaxis, :], X_obs, y_obs, eps)[0]
            
            try:
                result = minimize(
                    surrogate, x0, jac=surrogate_grad,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                new_x = np.clip(result.x, -5, 5)
            except Exception:
                # Fallback: random perturbation
                new_x = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
            
            # Evaluate the new point
            new_f = func(new_x)
            X_obs = np.vstack([X_obs, new_x])
            y_obs = np.append(y_obs, new_f)
            calls_made += 1
            
            # Update best
            if new_f < best_f:
                best_f = new_f
                best_x = new_x
            
            # Adapt epsilon based on number of points
            # As we have more points, we can use larger eps (more local)
            eps = min(2.0, 0.5 + 0.1 * len(X_obs) / 10)
            
            # If we have many points, also try exploring from other good points
            if len(X_obs) > 20 and calls_made < budget - 10:
                # Pick a random point that's not the best
                other_idx = np.random.choice([i for i in range(len(y_obs)) if i != idx_best])
                x0_alt = X_obs[other_idx].copy()
                
                try:
                    result_alt = minimize(
                        surrogate, x0_alt, jac=surrogate_grad,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 30, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    new_x_alt = np.clip(result_alt.x, -5, 5)
                    
                    # Only evaluate if it's different from recent points
                    if np.all(np.abs(new_x_alt - best_x) > 0.1):
                        new_f_alt = func(new_x_alt)
                        X_obs = np.vstack([X_obs, new_x_alt])
                        y_obs = np.append(y_obs, new_f_alt)
                        calls_made += 1
                        
                        if new_f_alt < best_f:
                            best_f = new_f_alt
                            best_x = new_x_alt
                except Exception:
                    pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
