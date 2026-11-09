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
        X1_scaled = X1 / length_scale
        X2_scaled = X2 / length_scale
        
        # Use broadcasting: (n1, 1, d) - (1, n2, d)
        diff = X1_scaled[:, np.newaxis, :] - X2_scaled[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        dist = np.sqrt(np.maximum(dist2, 0.0))
        
        # Matern 5/2
        return variance * (1 + np.sqrt(5) * dist + 5/3 * dist**2) * np.exp(-np.sqrt(5) * dist)

    def _fit_gp(self, X_train, y_train, initial_params=None):
        """Fit GP hyperparameters by maximizing log-likelihood using scipy minimize."""
        n = len(X_train)
        d = X_train.shape[1]
        
        # Normalize y for numerical stability
        y_mean = np.mean(y_train)
        y_std = max(np.std(y_train), 1e-8)
        y_norm = (y_train - y_mean) / y_std
        
        # Initialize hyperparameters
        if initial_params is None:
            length_scale = np.ones(d) * 1.5
            variance = 1.0
            noise_log = -4.0  # log(1e-4)
        else:
            length_scale, variance, noise_log = initial_params
        
        def compute_ll(params):
            """Compute log-likelihood for given hyperparameters."""
            ls = np.exp(params[:d])  # log-length-scale for positivity
            var = np.exp(params[d])
            nse = np.exp(params[d+1])
            
            K = self._kernel(X_train, X_train, ls, var) + nse * np.eye(n)
            try:
                L = np.linalg.cholesky(K + 1e-10 * np.eye(n))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                ll = -0.5 * y_norm @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
                return -ll  # Negate for minimization
            except:
                return 1e10
        
        # Initial parameters in log-space
        init_params = np.concatenate([
            np.log(length_scale),
            [np.log(variance)],
            [noise_log]
        ])
        
        # Bounds for log-space parameters
        bounds = [
            (np.log(0.1), np.log(10.0)) * d +
            [(np.log(0.01), np.log(100.0)), (np.log(1e-8), np.log(1e-2))]
        ]
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                compute_ll,
                init_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            opt_params = result.x
            length_scale = np.exp(opt_params[:d])
            variance = np.exp(opt_params[d])
            noise = np.exp(opt_params[d+1])
        except:
            # Fallback to initial values
            pass
        
        return length_scale, variance, noise, y_mean, y_std

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

    def _sobol_sample(self, n, dim, rng):
        """Generate quasi-random samples using a simple Sobol-like approach."""
        # Use Halton sequence for quasi-random points
        samples = np.zeros((n, dim))
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        for i in range(n):
            for d in range(dim):
                base = d + 2  # Prime bases: 2, 3, 5, 7, 11...
                samples[i, d] = halton(i + 1, base)
        
        # Scale to [-5, 5]
        samples = samples * 10 - 5
        return samples

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial sampling using quasi-random for better coverage
        n_init = min(30, budget // 10)
        X_list = []
        y_list = []

        # Quasi-random sampling for initial points
        X_init = self._sobol_sample(n_init, dim, rng)
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
        n_local_search = min(80, budget // 4)
        gp_budget = budget - n_init - n_local_search
        
        # Track for adaptive behavior
        no_improve_count = 0
        gp_improvements = 0
        
        initial_params = None  # For warm-starting GP fitting
        
        for gp_iter in range(min(20, gp_budget)):
            if calls_made >= budget:
                break

            # Fit GP
            try:
                length_scale, variance, noise, y_mean, y_std = self._fit_gp(X, y, initial_params)
                initial_params = (length_scale, variance, np.log(noise))  # Warm start
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
                    gp_improvements += 1
                else:
                    no_improve_count += 1
                continue

            # Optimize EI acquisition using multiple smart restarts
            n_restarts = 40
            best_acq = np.inf
            best_x_new = None

            # Generate diverse starting points
            restart_points = []
            restart_points.append(best_x.copy())  # Best point
            
            # Add points near existing data
            for i in range(min(10, len(X))):
                idx = rng.choice(len(X))
                x0 = np.clip(X[idx] + rng.normal(0, 0.5, dim), -5, 5)
                restart_points.append(x0)
            
            # Add random points
            for r in range(n_restarts - len(restart_points)):
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
                        options={'maxiter': 150, 'ftol': 1e-9}
                    )
                    if result.fun < best_acq:
                        best_acq = result.fun
                        best_x_new = np.clip(result.x, -5, 5)
                except:
                    continue

            if best_x_new is None:
                # Fallback: random point
                best_x_new = rng.uniform(-5, 5, dim)

            # Check if proposed point is too close to existing points
            min_dist = np.min(np.linalg.norm(X - best_x_new, axis=1))
            if min_dist < 0.05:
                # Add small random perturbation
                best_x_new = np.clip(best_x_new + rng.normal(0, 0.2, dim), -5, 5)

            # Evaluate new point
            y_new = func(best_x_new)
            X_list.append(best_x_new)
            y_list.append(y_new)
            calls_made += 1

            if y_new < best_y:
                best_x = best_x_new.copy()
                best_y = y_new
                no_improve_count = 0
                gp_improvements += 1
            else:
                no_improve_count += 1

            # Adaptive: if no improvement for a while, increase exploration
            if no_improve_count > 8:
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
                        gp_improvements += 1

            X = np.array(X_list)
            y = np.array(y_list)

        # Phase 3: Enhanced local search with adaptive step sizes and restarts
        x = best_x.copy()
        fx = best_y
        sigma = 0.2
        min_sigma = 0.005
        max_sigma = 1.5
        stuck_count = 0
        max_stuck = 15

        for i in range(n_local_search):
            if calls_made >= budget:
                break
            
            # If stuck, restart from best with larger sigma
            if stuck_count >= max_stuck:
                x = best_x.copy()
                fx = best_y
                sigma = 0.5
                stuck_count = 0
            
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.2, max_sigma)
                stuck_count = 0
                if fx < best_y:
                    best_x = x.copy()
                    best_y = fx
            else:
                sigma = max(sigma * 0.7, min_sigma)
                stuck_count += 1

        return best_x, best_y
# EVOLVE-BLOCK-END
