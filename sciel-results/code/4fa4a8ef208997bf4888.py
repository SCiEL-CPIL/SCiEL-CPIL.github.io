# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import qmc


class Optimizer:
    """Gaussian Process Bayesian Optimizer with Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _gp_predict(self, X_new, X, y, length_scale, noise):
        """
        Compute GP mean and variance at X_new using RBF kernel.
        """
        n = X.shape[0]
        
        # Compute kernel matrix K
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=-1)
        K = np.exp(-0.5 * sq_dist / (length_scale ** 2)) + noise * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-6 * np.eye(n)
            L = np.linalg.cholesky(K)

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        
        diff_new = X_new[:, np.newaxis, :] - X[np.newaxis, :, :]
        sq_dist_new = np.sum(diff_new ** 2, axis=-1)
        K_star = np.exp(-0.5 * sq_dist_new / (length_scale ** 2))
        
        mean = K_star @ alpha
        v = np.linalg.solve(L, K_star.T)
        var = 1.0 - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-10)
        
        return mean, var

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Initial points: Sobol sequence
        n_init = 20
        sampler = qmc.Sobol(d=dim, scramble=True)
        init_points = sampler.random(n_init)
        X = init_points * 10.0 - 5.0
        
        y = np.array([func(x) for x in X])
        
        # Track best
        y_best = np.min(y)
        x_best = X[np.argmin(y)]
        
        # GP Hyperparameters
        length_scale = 1.0
        noise = 1e-5
        max_gp_points = 100  # Limit GP size for speed
        
        # Bayesian optimization loop
        for i in range(n_init, budget):
            # Select subset of points for GP: Best points to focus on the valley
            if len(X) > max_gp_points:
                indices = np.argsort(y)[:max_gp_points]
                X_gp = X[indices]
                y_gp = y[indices]
            else:
                X_gp = X
                y_gp = y
            
            # Define Expected Improvement Acquisition Function
            def ei(x):
                x = np.array(x).reshape(1, -1)
                try:
                    mean, var = self._gp_predict(x, X_gp, y_gp, length_scale, noise)
                    std = np.sqrt(var[0])
                    if std < 1e-8:
                        return 0.0
                    z = (y_best - mean[0]) / std
                    ei_val = (y_best - mean[0]) * norm.cdf(z) + std * norm.pdf(z)
                    return -ei_val # Minimize negative EI
                except:
                    return 1e10

            # Optimize acquisition function
            # Start from multiple points: current best, random, local perturbations
            start_points = [x_best]
            start_points.extend(rng.uniform(-5, 5, (10, dim)))
            start_points.extend(x_best + 0.1 * rng.standard_normal((10, dim)))
            
            best_x = x_best
            best_ei = ei(x_best)
            
            for sp in start_points:
                sp = np.clip(sp, -5, 5)
                res = minimize(ei, sp, method='L-BFGS-B', bounds=[(-5, 5)] * dim, 
                               options={'maxiter': 50, 'ftol': 1e-4})
                if res.fun < best_ei:
                    best_ei = res.fun
                    best_x = res.x
            
            x_new = np.clip(best_x, -5, 5)
            
            # Evaluate function
            f_new = func(x_new)
            
            # Update data
            X = np.vstack([X, x_new])
            y = np.append(y, f_new)
            
            # Update best
            if f_new < y_best:
                y_best = f_new
                x_best = x_new
            
            # Optionally adapt length scale if we want, but fixed is often okay for Rosenbrock
            # Simple adaptation: if we haven't improved in a while, increase exploration
            # For now, keep fixed for stability and speed

        return x_best, y_best
# EVOLVE-BLOCK-END
