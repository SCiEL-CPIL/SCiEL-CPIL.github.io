"""Model-and-optimize optimizer for BBOB f19 (Griewank-Rosenbrock).

Uses a GP surrogate with PI-style acquisition for guided exploration,
then switches to local refinement with Nelder-Mead and L-BFGS-B.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from math import exp, sqrt, log

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _N_pdf(self, x, mu, sigma):
        """Standard normal PDF."""
        return exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * sqrt(2 * np.pi))

    def _N_cdf(self, x):
        """Standard normal CDF using erf approximation."""
        return 0.5 * (1.0 + np.tanh(0.7978845608028654 * x))

    def _fit_gp_simple(self, X, Y, noise=1e-6):
        """Fit a simple GP with RBF kernel and return params."""
        n = len(X)
        if n < 2:
            return None, None, None
        
        # Normalize inputs
        x_min = X.min(axis=0)
        x_max = X.max(axis=0)
        x_range = x_max - x_min
        x_range[x_range < 1e-8] = 1.0
        X_norm = (X - x_min) / x_range
        
        # Normalize outputs
        y_mean = Y.mean()
        y_std = Y.std()
        if y_std < 1e-8:
            y_std = 1.0
        Y_norm = (Y - y_mean) / y_std
        
        # RBF kernel
        def rbf_kernel(A, B, length_scale):
            dist_sq = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T
            dist_sq = np.maximum(dist_sq, 0)
            return exp(-dist_sq / (2 * length_scale**2))
        
        # Try a few length scales
        best_params = None
        best_ll = -np.inf
        for ls in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            K = rbf_kernel(X_norm, X_norm, ls) + noise * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
                ll = -0.5 * Y_norm @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * log(2 * np.pi)
                if ll > best_ll:
                    best_ll = ll
                    best_params = (ls, y_mean, y_std, x_min, x_range, alpha, L)
            except Exception:
                continue
        
        return best_params

    def _acq_pi(self, X_test, params, xi=0.01):
        """Probability of Improvement acquisition function."""
        if params is None:
            return np.zeros(len(X_test))
        
        ls, y_mean, y_std, x_min, x_range, alpha, L = params
        X_test_norm = (X_test - x_min) / x_range
        
        def rbf_kernel(A, B, length_scale):
            dist_sq = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T
            dist_sq = np.maximum(dist_sq, 0)
            return exp(-dist_sq / (2 * length_scale**2))
        
        # Get training data from params (reconstruct)
        # We need to store X and Y, so let's use a different approach
        # Actually, let's pass X and Y separately
        raise NotImplementedError("Use _acq_pi_with_data instead")

    def _acq_pi_with_data(self, X_test, X_train, Y_train, params, xi=0.01):
        """PI acquisition with explicit training data."""
        if params is None:
            return np.zeros(len(X_test))
        
        ls, y_mean, y_std, x_min, x_range, alpha, L = params
        X_test_norm = (X_test - x_min) / x_range
        X_train_norm = (X_train - x_min) / x_range
        
        def rbf_kernel(A, B, length_scale):
            dist_sq = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T
            dist_sq = np.maximum(dist_sq, 0)
            return exp(-dist_sq / (2 * length_scale**2))
        
        k = rbf_kernel(X_test_norm, X_train_norm, ls)
        mu = (k @ alpha) * y_std + y_mean
        v = rbf_kernel(X_test_norm, X_test_norm, ls)
        var = np.diag(v) - np.sum(k**2 * np.outer(alpha, alpha), axis=1)
        var = np.maximum(var, 1e-10)
        std = np.sqrt(var)
        
        f_best = Y_train.min()
        z = (f_best + xi - mu) / std
        pi = self._N_cdf(z)
        return pi

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        rng = np.random.default_rng(42)
        
        def elapsed():
            return time.time() - t0
        
        def sample(x):
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = float(func(x))
            return x, f
        
        # Phase 1: Initial exploration with diverse points
        X_data = []
        Y_data = []
        
        # Generate initial diverse points
        n_init = min(20, self.budget // 5)
        for i in range(n_init):
            if elapsed() > deadline - 10:
                break
            # Mix of uniform and LHS-like points
            if i == 0:
                x = np.zeros(self.dim)
            elif i == 1:
                x = rng.uniform(-4, 4, self.dim)
            else:
                # Use a structured grid with jitter
                base = (i / max(n_init - 2, 1)) * 8 - 4
                x = np.full(self.dim, base)
                x += rng.uniform(-0.5, 0.5, self.dim)
            
            x, f = sample(x)
            X_data.append(x)
            Y_data.append(f)
        
        best_x = X_data[0].copy()
        best_f = Y_data[0]
        for x, f in zip(X_data, Y_data):
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        calls_made = len(X_data)
        
        # Phase 2: GP-guided sequential model-based optimization
        gp_params = None
        stagnant_count = 0
        max_stagnant = 15
        
        while elapsed() < deadline - 20 and calls_made < self.budget - 10:
            # Fit GP model
            if len(X_data) >= 3:
                try:
                    gp_params = self._fit_gp_simple(np.array(X_data), np.array(Y_data))
                except Exception:
                    gp_params = None
            
            # Generate candidate points
            candidates = []
            
            # 1. Random exploration
            for _ in range(5):
                candidates.append(rng.uniform(-5, 5, self.dim))
            
            # 2. Local search around best
            for _ in range(5):
                x_cand = best_x + rng.normal(0, 0.3, self.dim)
                candidates.append(np.clip(x_cand, -5, 5))
            
            # 3. GP-guided points (if model available)
            if gp_params is not None:
                # Grid of points for acquisition
                grid_size = 8
                grid = np.array(np.meshgrid(*[np.linspace(-5, 5, grid_size)] * self.dim))
                grid = grid.reshape(self.dim, -1).T
                candidates.extend(grid.tolist())
            
            # Evaluate acquisition for each candidate
            acq_vals = []
            for cand in candidates:
                cand = np.clip(np.asarray(cand, dtype=float), -5, 5)
                if gp_params is not None:
                    try:
                        pi_val = self._acq_pi_with_data(
                            cand.reshape(1, -1),
                            np.array(X_data),
                            np.array(Y_data),
                            gp_params,
                            xi=0.01
                        )[0]
                    except Exception:
                        pi_val = 0.0
                else:
                    # Fallback: distance to best (prefer far points for exploration)
                    dist_to_best = np.linalg.norm(cand - best_x)
                    dist_to_all = min(np.linalg.norm(cand - x) for x in X_data)
                    pi_val = 0.5 * (1.0 / (1.0 + dist_to_best)) + 0.5 * (dist_to_all / 10.0)
                acq_vals.append(pi_val)
            
            # Select top candidates by acquisition value
            acq_vals = np.array(acq_vals)
            top_k = min(5, len(candidates))
            top_idx = np.argsort(-acq_vals)[:top_k]
            
            # Evaluate selected candidates
            for idx in top_idx:
                if elapsed() > deadline - 10 or calls_made >= self.budget - 10:
                    break
                x = np.clip(candidates[idx], -5, 5)
                x, f = sample(x)
                X_data.append(x)
                Y_data.append(f)
                calls_made += 1
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    stagnant_count = 0
                else:
                    stagnant_count += 1
            
            # If stagnant, add more random exploration
            if stagnant_count >= max_stagnant:
                for _ in range(3):
                    if elapsed() > deadline - 10 or calls_made >= self.budget - 10:
                        break
                    x = rng.uniform(-5, 5, self.dim)
                    x, f = sample(x)
                    X_data.append(x)
                    Y_data.append(f)
                    calls_made += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        stagnant_count = 0
        
        # Phase 3: Local polish with Nelder-Mead
        if elapsed() < deadline - 10:
            try:
                res = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass
        
        # Phase 4: L-BFGS-B polish
        if elapsed() < deadline - 5:
            try:
                res = minimize(
                    func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass
        
        # Final check
        if elapsed() < deadline - 1:
            try:
                _, f_final = sample(best_x)
                if f_final < best_f:
                    best_f = f_final
            except Exception:
                pass
        
        return best_x
# EVOLVE-BLOCK-END
