"""Surrogate-assisted Bayesian Optimization with GP and EI acquisition for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Surrogate-assisted Bayesian Optimization with Gaussian Process and EI."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _kernel(self, X1, X2, length_scale, variance):
        """Matern 5/2 kernel with ARD - vectorized computation."""
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        d = X1.shape[1]
        
        # Compute squared distances efficiently
        # X1: (n1, d), X2: (n2, d)
        # dist2[i,j] = sum_d (X1[i,d] - X2[j,d])^2 / ls[d]^2
        X1_scaled = X1 / length_scale
        X2_scaled = X2 / length_scale
        
        # Use broadcasting: (n1, 1, d) - (1, n2, d)
        diff = X1_scaled[:, np.newaxis, :] - X2_scaled[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        dist = np.sqrt(np.maximum(dist2, 0.0))
        
        # Matern 5/2
        return variance * (1 + np.sqrt(5) * dist + 5/3 * dist**2) * np.exp(-np.sqrt(5) * dist)

    def _fit_gp(self, X_train, y_train, initial_ls=None):
        """Fit GP hyperparameters by maximizing log-likelihood using coordinate ascent."""
        n = len(X_train)
        d = X_train.shape[1]
        
        # Normalize y for numerical stability
        y_mean = np.mean(y_train)
        y_std = max(np.std(y_train), 1e-8)
        y_norm = (y_train - y_mean) / y_std
        
        # Initialize hyperparameters
        if initial_ls is None:
            length_scale = np.ones(d) * 1.5
        else:
            length_scale = initial_ls.copy()
        variance = 1.0
        noise = 1e-4
        
        best_ll = -np.inf
        best_params = (length_scale.copy(), variance, noise)
        
        def compute_ll(ls, var, nse):
            """Compute log-likelihood for given hyperparameters."""
            K = self._kernel(X_train, X_train, ls, var) + nse * np.eye(n)
            try:
                L = np.linalg.cholesky(K + 1e-10 * np.eye(n))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                ll = -0.5 * y_norm @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
                return ll
            except:
                return -np.inf
        
        # Coordinate ascent for hyperparameters
        for iteration in range(8):
            # Optimize length scales (coordinate-wise)
            for i in range(d):
                best_ll_i = best_ll
                best_ls_i = length_scale[i]
                for mult in [0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.5, 4.0]:
                    ls_try = length_scale.copy()
                    ls_try[i] = best_ls_i * mult
                    ls_try = np.clip(ls_try, 0.1, 10.0)
                    ll = compute_ll(ls_try, variance, noise)
                    if ll > best_ll_i:
                        best_ll_i = ll
                        best_ls_i = ls_try[i]
                length_scale[i] = best_ls_i
            
            # Optimize variance
            best_ll_v = best_ll
            best_var = variance
            for mult in [0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.5]:
                var_try = variance * mult
                var_try = max(var_try, 0.01)
                ll = compute_ll(length_scale, var_try, noise)
                if ll > best_ll_v:
                    best_ll_v = ll
                    best_var = var_try
            variance = best_var
            
            # Optimize noise
            for nse_try in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
                ll = compute_ll(length_scale, variance, nse_try)
                if ll > best_ll:
                    best_ll = ll
                    noise = nse_try
            best_ll = compute_ll(length_scale, variance, noise)
            
            # Update best params
            if best_ll > best_params[3] if len(best_params) > 3 else best_ll > -np.inf:
                best_params = (length_scale.copy(), variance, noise)
        
        return best_params[0], best_params[1], best_params[2], y_mean, y_std

    def _gp_predict(self, X, X_train, y_train, length_scale, variance, noise, y_mean, y_std):
        """GP mean and variance prediction using Cholesky decomposition."""
        n = len(X_train)
        
        # Normalize training data
        y_norm = (y_train - y_mean) / y_std
        
        K = self._kernel(X_train, X_train, length_scale, variance) + noise * np.eye(n)
        try:
            L = np.linalg.cholesky(K + 1e-10 * np.eye(n))
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
        except:
            # Fallback to identity
            alpha = y_norm.copy()
            L = np.eye(n)
        
        K_star = self._kernel(X, X_train, length_scale, variance)
        mu_norm = K_star @ alpha
        mu = mu_norm * y_std + y_mean
        
        # Variance: var = K_ss - K_star @ K_inv @ K_star.T
        v = np.linalg.solve(L, K_star.T)
        var_norm = variance - np.sum(v ** 2, axis=1)
        var = np.maximum(var_norm, 0.0) * y_std ** 2
        
        return mu, np.sqrt(var)

    def _ei_acquisition(self, x, X_train, y_train, length_scale, variance, noise, y_mean, y_std, best_y):
        """Expected Improvement acquisition function (negative for minimization)."""
        x = x.reshape(1, -1)
        mu, sigma = self._gp_predict(x, X_train, y_train, length_scale, variance, noise, y_mean, y_std)
        mu = mu[0]
        sigma = sigma[0]
        if sigma < 1e-10:
            return 0.0
        z = (mu - best_y) / sigma
        ei = (best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return -ei  # Negate for minimization

    def _latin_hypercube_sample(self, n, dim, rng):
        """Generate Latin Hypercube samples in [-5, 5]^dim."""
        samples = np.zeros((n, dim))
        for d in range(dim):
            # Create n strata
            strata = np.linspace(0, 1, n + 1)
            # Random point in each stratum
            points = rng.uniform(strata[:-1], strata[1:])
            # Shuffle
            rng.shuffle(points)
            # Scale to [-5, 5]
            samples[:, d] = points * 10 - 5
        return samples

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial sampling using Latin Hypercube for better coverage
        n_init = min(40, budget // 8)
        X_list = []
        y_list = []

        # Latin Hypercube sampling for initial points
        X_init = self._latin_hypercube_sample(n_init, dim, rng)
        for i in range(n_init):
            x = X_init[i]
            y = func(x)
            X_list.append(x)
            y_list.append(y)

        X = np.array(X_list)
        y = np.array(y_list)
        best_idx = np.argmin(y)
        best_x = X[best_idx].copy()
        best_y = y[best_idx]

        calls_made = n_init

        # Phase 2: GP-based Bayesian Optimization
        # Reserve budget for local search
        n_local_search = min(60, budget // 5)
        gp_budget = budget - n_init - n_local_search
        
        # Track for adaptive behavior
        last_best_y = best_y
        no_improve_count = 0
        
        initial_ls = None  # For warm-starting GP fitting
        
        for gp_iter in range(min(15, gp_budget)):
            if calls_made >= budget:
                break

            # Fit GP
            try:
                length_scale, variance, noise, y_mean, y_std = self._fit_gp(X, y, initial_ls)
                initial_ls = length_scale  # Warm start for next iteration
            except:
                # Fallback: use current best with small perturbation
                x_new = np.clip(best_x + rng.normal(0, 0.3, dim), -5, 5)
                y_new = func(x_new)
                X_list.append(x_new)
                y_list.append(y_new)
                calls_made += 1
                if y_new < best_y:
                    best_x = x_new.copy()
                    best_y = y_new
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                continue

            # Optimize EI acquisition using multiple smart restarts
            n_restarts = 25
            best_acq = np.inf
            best_x_new = None
            best_acq_val = np.inf

            # Generate diverse starting points
            restart_points = []
            restart_points.append(best_x.copy())  # Best point
            restart_points.append(rng.uniform(-5, 5, dim))  # Random
            for r in range(n_restarts - 2):
                # Mix of perturbations around best and random
                if r % 3 == 0:
                    # Small perturbation around best
                    x0 = np.clip(best_x + rng.normal(0, 0.5, dim), -5, 5)
                elif r % 3 == 1:
                    # Medium perturbation
                    x0 = np.clip(best_x + rng.normal(0, 1.5, dim), -5, 5)
                else:
                    # Random in box
                    x0 = rng.uniform(-5, 5, dim)
                restart_points.append(x0)

            for r, x0 in enumerate(restart_points):
                try:
                    result = minimize(
                        self._ei_acquisition,
                        x0,
                        args=(X, y, length_scale, variance, noise, y_mean, y_std, best_y),
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 100, 'ftol': 1e-8}
                    )
                    if result.fun < best_acq:
                        best_acq = result.fun
                        best_x_new = np.clip(result.x, -5, 5)
                        best_acq_val = result.fun
                except:
                    continue

            if best_x_new is None:
                # Fallback: random point
                best_x_new = rng.uniform(-5, 5, dim)

            # Check if proposed point is too close to existing points
            min_dist = np.min(np.linalg.norm(X - best_x_new, axis=1))
            if min_dist < 0.01:
                # Add small random perturbation
                best_x_new = np.clip(best_x_new + rng.normal(0, 0.1, dim), -5, 5)

            # Evaluate new point
            y_new = func(best_x_new)
            X_list.append(best_x_new)
            y_list.append(y_new)
            calls_made += 1

            if y_new < best_y:
                best_x = best_x_new.copy()
                best_y = y_new
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Adaptive: if no improvement for a while, increase exploration
            if no_improve_count > 5:
                # Inject a random point to escape local optimum
                if calls_made < budget - 2:
                    x_rand = rng.uniform(-5, 5, dim)
                    y_rand = func(x_rand)
                    X_list.append(x_rand)
                    y_list.append(y_rand)
                    calls_made += 1
                    if y_rand < best_y:
                        best_x = x_rand.copy()
                        best_y = y_rand
                        no_improve_count = 0

            X = np.array(X_list)
            y = np.array(y_list)

        # Phase 3: Local (1+1)-ES around best point with adaptive sigma
        x = best_x.copy()
        fx = best_y
        sigma = 0.15
        min_sigma = 0.01
        max_sigma = 1.0

        for i in range(n_local_search):
            if calls_made >= budget:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.15, max_sigma)
                if fx < best_y:
                    best_x = x.copy()
                    best_y = fx
            else:
                sigma = max(sigma * 0.8, min_sigma)

        return best_x, best_y
# EVOLVE-BLOCK-END
