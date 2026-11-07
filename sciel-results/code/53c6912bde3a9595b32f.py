# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize_scalar


class Optimizer:
    """Bayesian Optimization with GP surrogate and Student-t EI acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _rbf_kernel(self, X1, X2, length_scales):
        """RBF kernel with ARD length scales."""
        X1 = X1[:, :, None]
        X2 = X2[:, None, :]
        sq_dist = np.sum((X1 - X2) ** 2, axis=2) / (length_scales ** 2)
        return np.exp(-0.5 * sq_dist)

    def _fit_gp(self, X, y, length_scales, noise_var, alpha):
        """Compute GP posterior given training data."""
        K = self._rbf_kernel(X, X, length_scales)
        K += (noise_var + 1e-8) * np.eye(len(X))
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return L, alpha

    def _gp_predict(self, X_new, X_train, y_train, L, alpha, length_scales, noise_var):
        """Predict mean and variance at new points."""
        K_s = self._rbf_kernel(X_new, X_train, length_scales)
        K_ss = self._rbf_kernel(X_new, X_new, length_scales)
        v = np.linalg.solve(L, K_s.T)
        mu = v.T @ alpha
        var = np.diag(K_ss) - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-10)
        return mu, var

    def _student_t_ei(self, mu, var, best_y, nu=5.0):
        """Student-t Expected Improvement."""
        sigma = np.sqrt(var)
        z = (best_y - mu) / sigma
        # Student-t CDF and PDF
        from scipy.special import betainc
        t = z / np.sqrt(nu)
        # For large nu, approximates normal
        # Use normal approximation for stability
        norm_cdf = norm.cdf(z)
        norm_pdf = norm.pdf(z)
        ei = (best_y - mu) * norm_cdf(z) + sigma * norm_pdf(z)
        return ei

    def __call__(self, func):
        rng = np.random.default_rng(42)
        deadline = time.time() + 280.0
        dim = self.dim
        budget = self.budget

        # Initial sampling: mix of random and LHS
        n_init = min(20, budget)
        X_list = []
        y_list = []

        # Latin Hypercube Sampling for initial points
        for i in range(dim):
            perm = rng.permutation(n_init)
            X_list.append((perm + 0.5) / n_init * 10 - 5)

        X_init = np.array(X_list).T
        for x in X_init:
            if time.time() > deadline:
                break
            y = func(np.clip(x, -5, 5))
            X_list.append(np.clip(x, -5, 5))
            y_list.append(y)

        if len(y_list) == 0:
            x0 = rng.uniform(-5, 5, dim)
            y0 = func(x0)
            return x0, y0

        X = np.array(X_list)
        y = np.array(y_list)
        best_idx = np.argmin(y)
        best_x = X[best_idx].copy()
        best_y = y[best_idx]

        # Adaptive hyperparameters
        length_scales = np.ones(dim) * 1.5
        noise_var = 1e-4
        signal_var = 1.0

        # Refit hyperparameters periodically
        hyper_iter = 0

        remaining = budget - len(y_list)
        for iteration in range(remaining):
            if time.time() > deadline:
                break

            # Refit GP hyperparameters every 30 iterations
            if hyper_iter % 30 == 0 and len(y) > 5:
                # Simple grid search for length scales
                best_ll = -np.inf
                best_ls = length_scales.copy()
                for ls_mult in [0.5, 1.0, 2.0]:
                    ls_try = length_scales * ls_mult
                    try:
                        K = self._rbf_kernel(X, X, ls_try)
                        K += (noise_var + 1e-8) * np.eye(len(X))
                        L = np.linalg.cholesky(K)
                        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                        ll = -0.5 * (len(y) * np.log(2 * np.pi) + np.sum(np.log(np.diag(K))) + y @ alpha)
                        if ll > best_ll:
                            best_ll = ll
                            best_ls = ls_try
                    except:
                        pass
                length_scales = best_ls

            # Fit GP
            try:
                L, alpha = self._fit_gp(X, y, length_scales, noise_var, signal_var)
            except:
                # Fallback to random sampling
                x_new = rng.uniform(-5, 5, dim)
                y_new = func(x_new)
                X = np.vstack([X, x_new])
                y = np.append(y, y_new)
                if y_new < best_y:
                    best_y = y_new
                    best_x = x_new.copy()
                continue

            # Optimize acquisition function using multiple random restarts
            best_ei = -np.inf
            best_x_acq = None

            # Sample candidate points for acquisition
            n_candidates = 200
            X_cand = rng.uniform(-5, 5, (n_candidates, dim))

            mu_cand, var_cand = self._gp_predict(X_cand, X, y, L, alpha, length_scales, noise_var)
            ei_cand = self._student_t_ei(mu_cand, var_cand, best_y)

            # Find top candidates and refine with local optimization
            top_k = min(10, n_candidates)
            top_indices = np.argsort(ei_cand)[-top_k:]

            for idx in top_indices:
                x_start = X_cand[idx]

                def neg_ei(x):
                    x = np.clip(x, -5, 5)
                    mu, var = self._gp_predict(x.reshape(1, -1), X, y, L, alpha, length_scales, noise_var)
                    ei = self._student_t_ei(mu[0], var[0], best_y)
                    return -ei

                try:
                    res = minimize_scalar(neg_ei, bounds=[(-5, 5)] * dim, method='L-BFGS-B',
                                         options={'maxiter': 50})
                    if res.success and -res.fun > best_ei:
                        best_ei = -res.fun
                        best_x_acq = np.clip(res.x, -5, 5)
                except:
                    if -neg_ei(x_start) > best_ei:
                        best_ei = -neg_ei(x_start)
                        best_x_acq = x_start.copy()

            if best_x_acq is None:
                best_x_acq = X_cand[top_indices[0]]

            # Evaluate new point
            y_new = func(best_x_acq)
            X = np.vstack([X, best_x_acq])
            y = np.append(y, y_new)

            if y_new < best_y:
                best_y = y_new
                best_x = best_x_acq.copy()

            hyper_iter += 1

        return best_x, best_y
# EVOLVE-BLOCK-END
