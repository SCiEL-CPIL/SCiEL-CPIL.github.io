"""Rewritten program: GP-based Bayesian optimization with Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


class Optimizer:
    """Bayesian optimizer with Gaussian Process surrogate and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        B = self.budget

        # Reserve last ~30% for local refinement
        n_local = max(10, int(B * 0.3))
        n_bo = B - n_local

        # Initial sampling: mix of random and structured
        n_init = min(15, n_bo)
        X = np.zeros((n_init, d))
        fvals = np.zeros(n_init)
        
        # First point: center
        X[0] = np.zeros(d)
        fvals[0] = func(X[0])
        
        # Rest: random
        for i in range(1, n_init):
            X[i] = rng.uniform(-5, 5, d)
            fvals[i] = func(X[i])

        # Gaussian Process kernel: Matern 5/2
        def matern52_kernel(X1, X2, length_scale, variance):
            D = cdist(X1, X2)
            return variance * (1 + np.sqrt(5) * D / length_scale) * np.exp(-np.sqrt(5) * D / length_scale)

        def log_likelihood(theta, X, y):
            ls, var, noise = np.exp(theta)
            K = matern52_kernel(X, X, ls, var) + noise * np.eye(len(X))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return -1e10
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            ll = -0.5 * y @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * len(y) * np.log(2 * np.pi)
            return ll

        # Optimize kernel parameters once at start
        theta_init = np.array([np.log(1.0), np.log(1.0), np.log(0.01)])
        try:
            from scipy.optimize import minimize as sp_minimize
            res = sp_minimize(
                lambda t: -log_likelihood(t, X, fvals),
                theta_init,
                method='L-BFGS-B',
                bounds=[(-2, 2), (-1, 3), (-4, 0)]
            )
            theta_opt = res.x
        except Exception:
            theta_opt = theta_init

        ls, var, noise = np.exp(theta_opt)

        def gp_mean_var(X_new, X_obs, y_obs):
            K = matern52_kernel(X_new, X_obs, ls, var)
            K_ss = matern52_kernel(X_obs, X_obs, ls, var) + noise * np.eye(len(X_obs))
            try:
                K_ss_inv = np.linalg.inv(K_ss)
            except np.linalg.LinAlgError:
                K_ss_inv = np.linalg.pinv(K_ss)
            K_star = K.T
            mean = K_star @ K_ss_inv @ y_obs
            K_ss_new = matern52_kernel(X_new, X_new, ls, var)
            cov = K_ss_new - K_star @ K_ss_inv @ K
            var_arr = np.diag(cov)
            var_arr = np.maximum(var_arr, 1e-10)
            return mean, var_arr

        def expected_improvement(x, mu, sigma, f_min):
            sigma = np.maximum(sigma, 1e-8)
            z = (f_min - mu) / sigma
            from scipy.stats import norm
            EI = (f_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return EI

        best_x = X[np.argmin(fvals)]
        best_f = np.min(fvals)

        # Bayesian optimization loop
        for i in range(n_init, n_bo):
            # Candidate points for EI maximization
            n_cand = 200
            candidates = rng.uniform(-5, 5, (n_cand, d))
            
            mu, var_arr = gp_mean_var(candidates, X, fvals)
            EI = expected_improvement(candidates, mu, var_arr, best_f)
            
            # Add some random exploration
            idx = np.argmax(EI)
            x_new = candidates[idx]
            
            f_new = func(x_new)
            X = np.vstack([X, x_new])
            fvals = np.append(fvals, f_new)
            
            if f_new < best_f:
                best_f = f_new
                best_x = x_new

        # Local refinement with Nelder-Mead
        # Use the best point found so far as starting point
        x0 = best_x.copy()
        
        # Try a few different starting points for NM
        start_points = [best_x]
        # Add a few nearby points
        for _ in range(3):
            perturbation = rng.uniform(-0.5, 0.5, d)
            sp = np.clip(best_x + perturbation, -5, 5)
            start_points.append(sp)

        final_x = best_x
        final_f = best_f

        for sp in start_points:
            try:
                result = minimize(
                    func,
                    sp,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6}
                )
                if result.fun < final_f:
                    final_f = result.fun
                    final_x = result.x
            except Exception:
                pass

        return final_x, final_f
# EVOLVE-BLOCK-END
