"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class Optimizer:
    """Surrogate-based optimizer using a Gaussian Process model
       with expected improvement acquisition function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Storage for observations
        X = []
        Y = []

        def add_obs(x, y):
            X.append(np.asarray(x, dtype=float))
            Y.append(float(y))

        # Initial sampling: diverse points to build initial GP model
        n_init = min(20, budget // 5)
        for i in range(n_init):
            if i == 0:
                x0 = np.zeros(dim)
            elif i == 1:
                x0 = np.full(dim, 2.0)
            elif i == 2:
                x0 = np.full(dim, -2.0)
            else:
                x0 = rng.uniform(-5, 5, dim)
            fx = func(x0)
            add_obs(x0, fx)

        # Track best
        best_x = X[0].copy()
        best_f = Y[0]
        for i in range(1, len(X)):
            if Y[i] < best_f:
                best_f = Y[i]
                best_x = X[i].copy()

        # GP hyperparameters
        length_scale = 1.5
        variance = 1.0
        noise = 1e-6

        def compute_K(X1, X2, ls):
            """Compute Gaussian RBF kernel matrix."""
            d2 = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            d2 = np.maximum(d2, 0)
            return np.exp(-d2 / (2 * ls**2))

        def gp_predict(x_new, X_train, Y_train, ls, var, noise):
            """
            Predict mean and std at x_new using GP with RBF kernel.
            Returns (mu, sigma).
            """
            X_arr = np.array(X_train)
            Y_arr = np.array(Y_train)
            n = len(X_train)
            
            if n == 0:
                return np.mean(Y_arr) if len(Y_arr) > 0 else 0.0, var

            # Kernel matrix K = K(X_train, X_train)
            K = compute_K(X_arr, X_arr, ls)
            K += noise * np.eye(n)
            
            # Kernel vector k = K(X_train, x_new)
            x_new_arr = x_new.reshape(1, -1)
            k = compute_K(X_arr, x_new_arr, ls).flatten()
            
            try:
                # Solve for alpha
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_arr))
                
                # Predicted mean
                mu = k @ alpha
                
                # Predicted variance
                v = np.linalg.solve(L, k)
                sigma2 = var - k @ v
                sigma2 = max(sigma2, 1e-10)
                sigma = np.sqrt(sigma2)
            except np.linalg.LinAlgError:
                # Fallback: use simple weighted average
                d2 = np.sum((X_arr - x_new)**2, axis=1)
                K = np.exp(-d2 / (2 * ls**2))
                if np.sum(K) < 1e-12:
                    mu = np.mean(Y_arr)
                    sigma = np.std(Y_arr) + 0.5
                else:
                    mu = np.sum(K * Y_arr) / np.sum(K)
                    sigma = np.std(Y_arr) * (1 - np.max(K)) + 0.1
                    sigma = max(sigma, 0.01)
            
            return mu, sigma

        def acquisition_ei(x):
            """
            Expected Improvement acquisition function.
            EI(x) = (mu_best - mu) * Phi(z) + sigma * phi(z)
            where z = (mu_best - mu) / sigma
            """
            mu, sigma = gp_predict(x, X, Y, length_scale, variance, noise)
            
            if sigma < 1e-10:
                return max(0, best_f - mu)
            
            z = (best_f - mu) / sigma
            # Standard normal CDF and PDF
            Phi = 0.5 * (1 + np.erf(z / np.sqrt(2)))
            phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
            
            ei = (best_f - mu) * Phi + sigma * phi
            return ei

        # Optimization loop
        calls_made = len(X)
        
        # Adaptive parameters
        explore_weight = 1.0  # Will be adjusted
        
        for step in range(budget - calls_made):
            # Select next point by optimizing acquisition function
            # Use multiple random starts for local optimization
            n_starts = 8
            best_ei = -np.inf
            best_x_next = None

            # Candidate starts
            starts = []
            # Random points
            for _ in range(4):
                starts.append(rng.uniform(-5, 5, dim))
            # Current best
            starts.append(best_x.copy())
            # Points near best (exploit)
            for _ in range(2):
                starts.append(np.clip(best_x + rng.normal(0, 0.3, dim), -5, 5))
            # Points in between (balance)
            for _ in range(2):
                starts.append(np.clip(best_x * 0.5 + rng.uniform(-2, 2, dim), -5, 5))

            for start in starts:
                start = np.clip(start, -5, 5)
                try:
                    res = minimize(
                        lambda z: -acquisition_ei(z),
                        start,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 30, 'ftol': 1e-6}
                    )
                    ei_val = -res.fun
                    if ei_val > best_ei:
                        best_ei = ei_val
                        best_x_next = np.clip(res.x, -5, 5)
                except:
                    continue

            if best_x_next is None:
                best_x_next = rng.uniform(-5, 5, dim)

            # Evaluate the function
            f_new = func(best_x_next)
            add_obs(best_x_next, f_new)
            calls_made += 1

            # Update best
            if f_new < best_f:
                best_f = f_new
                best_x = best_x_next.copy()

            # Adapt length scale based on progress and data spread
            if len(X) >= 10:
                X_arr = np.array(X)
                Y_arr = np.array(Y)
                
                # Estimate typical improvement scale
                improvements = np.diff(np.sort(Y_arr))
                if len(improvements) > 0:
                    typical_improve = np.median(improvements[improvements > 0]) if np.any(improvements > 0) else 0.1
                
                # Adjust length scale: smaller if making good progress, larger if stuck
                if best_f < np.mean(Y_arr) - 0.1:
                    # Making progress, can explore more finely
                    length_scale = max(0.3, length_scale * 0.98)
                else:
                    # Stuck, explore more broadly
                    length_scale = min(3.0, length_scale * 1.02)
                
                # Also adapt based on data spread
                if len(X_arr) <= 50:
                    idx = rng.choice(len(X_arr), size=min(20, len(X_arr)), replace=False)
                    X_sub = X_arr[idx]
                    d2 = np.sum((X_sub[:, None, :] - X_sub[None, :, :])**2, axis=2)
                    np.fill_diagonal(d2, np.inf)
                    med_dist = np.median(d2)
                    if med_dist > 1e-10:
                        target_ls = np.sqrt(med_dist) * 0.5
                        length_scale = 0.7 * length_scale + 0.3 * max(0.3, min(3.0, target_ls))

            # Adapt variance
            if len(X) >= 5:
                Y_arr = np.array(Y)
                var_est = max(np.var(Y_arr), 0.1)
                variance = 0.8 * variance + 0.2 * var_est

        return best_x, best_f
# EVOLVE-BLOCK-END
