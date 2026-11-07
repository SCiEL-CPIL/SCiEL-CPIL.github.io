"""Model-and-optimize optimizer for BBOB f19 (Griewank-Rosenbrock).

Combines a (1+1)-ES exploration phase with a GP surrogate model (model-and-optimize)
using PI acquisition, followed by Nelder-Mead and L-BFGS-B local polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from math import exp, sqrt, log
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _N_pdf(self, z):
        return exp(-0.5 * z * z) / sqrt(2.0 * np.pi)

    def _N_cdf(self, z):
        # Abramowitz-Stegun approximation
        t = 1.0 / (1.0 + 0.2316419 * abs(z))
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        cdf = 1.0 - poly * exp(-0.5 * z * z) / sqrt(2.0 * np.pi)
        return 0.5 * (1.0 + np.sign(z) * cdf) if z != 0 else 0.5

    def _fit_gp(self, X, y):
        """Fit a GP with RBF kernel and return (alpha, b, l2, X, y)."""
        n = len(X)
        if n < 2:
            return None
        
        # Normalize inputs
        x_min = X.min(axis=0)
        x_max = X.max(axis=0)
        x_range = x_max - x_min
        x_range[x_range < 1e-8] = 1.0
        X_norm = (X - x_min) / x_range
        
        # Normalize outputs
        y_mean = float(y.mean())
        y_std = float(y.std())
        if y_std < 1e-8:
            y_std = 1.0
        Y_norm = (y - y_mean) / y_std
        
        # Try a few length scales
        best_params = None
        best_ll = -np.inf
        
        for ls in [0.2, 0.4, 0.6, 0.8, 1.0, 1.5]:
            # RBF kernel
            dist_sq = np.sum(X_norm**2, axis=1)[:, None] + np.sum(X_norm**2, axis=1)[None, :] - 2 * X_norm @ X_norm.T
            dist_sq = np.maximum(dist_sq, 0)
            K = exp(-dist_sq / (2 * ls**2))
            K[np.arange(n), np.arange(n)] += 1e-6
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L, Y_norm)
                ll = -0.5 * Y_norm @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * log(2 * np.pi)
                if ll > best_ll:
                    best_ll = ll
                    best_params = (alpha, y_mean, y_std, x_min, x_range, ls, L)
            except Exception:
                continue
        
        return best_params

    def _acq_pi(self, X_test, X_train, Y_train, params, xi=0.01):
        """Probability of Improvement acquisition function."""
        if params is None:
            return np.zeros(len(X_test))
        
        alpha, y_mean, y_std, x_min, x_range, ls, L = params
        X_test_norm = (X_test - x_min) / x_range
        X_train_norm = (X_train - x_min) / x_range
        
        # RBF kernel
        dist_sq = np.sum(X_test_norm**2, axis=1)[:, None] + np.sum(X_train_norm**2, axis=1)[None, :] - 2 * X_test_norm @ X_train_norm.T
        dist_sq = np.maximum(dist_sq, 0)
        k = exp(-dist_sq / (2 * ls**2))
        
        mu = (k @ alpha) * y_std + y_mean
        v = exp(-np.sum(X_test_norm**2, axis=1) / ls**2) * y_std**2
        var = np.maximum(v - np.sum(k**2 * np.outer(alpha, alpha), axis=1) * y_std**2, 1e-10)
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
        
        # Phase 1: Initial exploration with (1+1)-ES
        X_data = []
        Y_data = []
        
        # Start with a random point
        x0 = rng.uniform(-4, 4, self.dim)
        x0, f0 = sample(x0)
        X_data.append(x0)
        Y_data.append(f0)
        best_x = x0.copy()
        best_f = f0
        calls_made = 1
        
        # (1+1)-ES phase
        x_es = best_x.copy()
        f_es = best_f
        sigma = 1.5
        n_es = min(15, self.budget // 10)
        
        for _ in range(n_es):
            if elapsed() > deadline - 30 or calls_made >= self.budget - 20:
                break
            y = np.clip(x_es + sigma * rng.standard_normal(self.dim), -5, 5)
            y, fy = sample(y)
            X_data.append(y)
            Y_data.append(fy)
            calls_made += 1
            
            if fy < f_es:
                x_es, f_es = y, fy
                sigma *= 1.1
            else:
                sigma *= 0.85
            
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
        
        # Phase 2: GP-guided sequential model-based optimization
        stagnant_count = 0
        max_stagnant = 10
        
        while elapsed() < deadline - 40 and calls_made < self.budget - 30:
            # Fit GP model
            gp_params = None
            if len(X_data) >= 4:
                try:
                    gp_params = self._fit_gp(np.array(X_data), np.array(Y_data))
                except Exception:
                    gp_params = None
            
            # Generate candidate points
            candidates = []
            
            # 1. Random exploration
            for _ in range(4):
                candidates.append(rng.uniform(-5, 5, self.dim))
            
            # 2. Local search around best
            for _ in range(4):
                x_cand = best_x + rng.normal(0, 0.4, self.dim)
                candidates.append(np.clip(x_cand, -5, 5))
            
            # 3. GP-guided points (if model available)
            if gp_params is not None:
                # Grid of points for acquisition
                grid_size = 7
                grid = np.array(np.meshgrid(*[np.linspace(-5, 5, grid_size)] * self.dim))
                grid = grid.reshape(self.dim, -1).T
                candidates.extend(grid.tolist())
            
            # Evaluate acquisition for each candidate
            acq_vals = []
            for cand in candidates:
                cand = np.clip(np.asarray(cand, dtype=float), -5, 5)
                if gp_params is not None:
                    try:
                        pi_val = self._acq_pi(
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
            top_k = min(4, len(candidates))
            top_idx = np.argsort(-acq_vals)[:top_k]
            
            # Evaluate selected candidates
            for idx in top_idx:
                if elapsed() > deadline - 30 or calls_made >= self.budget - 30:
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
                    if elapsed() > deadline - 30 or calls_made >= self.budget - 30:
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
        if elapsed() < deadline - 15:
            try:
                res = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
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
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
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
