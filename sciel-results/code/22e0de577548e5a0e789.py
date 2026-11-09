"""Bayesian Optimization with GP surrogate and EI acquisition for Katsuura f23."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """GP-based Bayesian Optimization for Katsuura f23 on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _rbf_kernel(self, X1, X2, length_scales, variance):
        """Compute RBF kernel matrix between X1 and X2."""
        # X1: (n1, d), X2: (n2, d)
        # length_scales: (d,)
        sq_dists = (
            np.sum(X1 ** 2, axis=1)[:, None]
            + np.sum(X2 ** 2, axis=1)[None, :]
            - 2 * X1 @ X2.T
        )
        sq_dists = np.maximum(sq_dists, 0)
        # Scale by length scales
        sq_dists = sq_dists / (length_scales ** 2)
        K = variance * np.exp(-0.5 * sq_dists)
        return K

    def _gp_predict(self, X, X_train, y_train, length_scales, variance, noise):
        """Predict mean and variance at X using GP with RBF kernel."""
        K_train = self._rbf_kernel(X_train, X_train, length_scales, variance)
        K_train += noise * np.eye(len(X_train))
        K_cross = self._rbf_kernel(X, X_train, length_scales, variance)
        
        try:
            L = np.linalg.cholesky(K_train + 1e-10 * np.eye(len(X_train)))
        except np.linalg.LinAlgError:
            K_train += 1e-6 * np.eye(len(X_train))
            L = np.linalg.cholesky(K_train + 1e-10 * np.eye(len(X_train)))
        
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        mean = K_cross @ alpha
        v = np.linalg.solve(L, K_cross.T)
        var = variance - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-12)
        return mean, var

    def _ei_acquisition(self, X, X_train, y_train, length_scales, variance, noise, best_f, xi=0.01):
        """Expected Improvement acquisition function."""
        mean, var = self._gp_predict(X, X_train, y_train, length_scales, variance, noise)
        std = np.sqrt(var)
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = (best_f - mean - xi) / std
            Z = np.where(std > 1e-12, Z, 0)
            from scipy.stats import norm
            EI = (best_f - mean - xi) * norm.cdf(Z) + std * norm.pdf(Z)
            EI = np.where(std > 1e-12, EI, 0)
        return EI

    def _optimize_acquisition(self, X_train, y_train, length_scales, variance, noise, best_f, rng, n_restarts=10):
        """Optimize EI acquisition function using multi-start L-BFGS."""
        dim = self.dim
        best_x = None
        best_ei = -np.inf
        
        # Initial candidates: random points + around best known
        candidates = []
        for _ in range(n_restarts):
            x = rng.uniform(-5, 5, size=dim)
            candidates.append(x)
        
        # Also add perturbations around best point
        if len(X_train) > 0:
            best_idx = np.argmin(y_train)
            best_pt = X_train[best_idx]
            for _ in range(5):
                x = best_pt + rng.normal(0, 0.5, size=dim)
                x = np.clip(x, -5, 5)
                candidates.append(x)
        
        for x0 in candidates:
            def neg_ei(x):
                x = np.clip(x, -5, 5)
                ei = self._ei_acquisition(
                    x.reshape(1, -1), X_train, y_train,
                    length_scales, variance, noise, best_f
                )
                return -ei[0]
            
            try:
                res = minimize(neg_ei, x0, method='L-BFGS-B',
                              bounds=[(-5, 5)] * dim,
                              options={'maxiter': 50})
                if -res.fun > best_ei:
                    best_ei = -res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                continue
        
        if best_x is None:
            best_x = np.clip(candidates[0], -5, 5)
        
        return best_x

    def _latin_hypercube(self, rng, n_samples, dim):
        """Generate n_samples points via Latin Hypercube Sampling in [-5, 5]^dim."""
        points = np.zeros((n_samples, dim))
        for d in range(dim):
            perm = rng.permutation(n_samples)
            points[:, d] = (perm + 0.5) / n_samples * 10.0 - 5.0
        return points

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285.0

        # Phase 1: Latin Hypercube initialization
        n_init = 25
        lhd = self._latin_hypercube(rng, n_init, dim)
        
        X_train = []
        y_train = []
        calls = 0
        
        for i in range(n_init):
            if time.time() > deadline or calls >= budget:
                break
            x = lhd[i]
            f = func(x)
            calls += 1
            X_train.append(x)
            y_train.append(f)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        best_f = np.min(y_train)
        best_x = X_train[np.argmin(y_train)].copy()

        # Phase 2: Bayesian Optimization loop
        # GP hyperparameters
        variance = 1.0
        noise = 1e-6
        length_scales = np.ones(dim) * 1.0
        
        # Adaptive length scales based on data spread
        if len(X_train) > 1:
            for d in range(dim):
                spread = np.ptp(X_train[:, d])
                if spread > 1e-8:
                    length_scales[d] = spread / 3.0
            length_scales = np.clip(length_scales, 0.1, 5.0)
        
        remaining = budget - calls
        
        for i in range(remaining):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            # Update GP hyperparameters periodically
            if i > 0 and i % 20 == 0 and len(X_train) > 5:
                # Estimate variance from data
                y_std = np.std(y_train)
                if y_std > 1e-10:
                    variance = max(y_std ** 2, 0.01)
                # Update length scales
                for d in range(dim):
                    spread = np.ptp(X_train[:, d])
                    if spread > 1e-8:
                        length_scales[d] = max(spread / 3.0, 0.1)
            
            # Select next point using EI
            x_next = self._optimize_acquisition(
                X_train, y_train, length_scales, variance, noise, best_f, rng
            )
            
            # Evaluate
            f_next = func(x_next)
            calls += 1
            
            # Update training data
            X_train = np.vstack([X_train, x_next.reshape(1, -1)])
            y_train = np.append(y_train, f_next)
            
            # Update best
            if f_next < best_f:
                best_f = f_next
                best_x = x_next.copy()
        
        return best_x, best_f
