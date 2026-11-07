import numpy as np
import time
import sys
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None
        self.time_limit = 300.0
        self.X = []
        self.Y = []
        self.L = None
        self.alpha = None
        # Fixed hyperparameters for robustness
        self.sigma = 1.0
        self.lengthscales = np.ones(dim) * 1.0

    def _check_time(self):
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) > (self.time_limit - 10.0)

    def _evaluate(self, x):
        if self.calls >= self.budget:
            return None
        if self._check_time():
            return None
        x = np.asarray(x, dtype=float)
        x = np.clip(x, -5.0, 5.0)
        self.calls += 1
        try:
            val = float(self.func(x))
        except Exception:
            val = float('inf')
        if val < self.best_f:
            self.best_f = val
            self.best_x = x.copy()
        return val

    def _rbf_kernel(self, X1, X2):
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        X1_norm = np.sum(X1**2, axis=1, keepdims=True)
        X2_norm = np.sum(X2**2, axis=1, keepdims=True).T
        dist_sq = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
        dist_sq = np.maximum(dist_sq, 0) # Numerical stability
        dist_sq = dist_sq / (self.lengthscales**2)
        return self.sigma**2 * np.exp(-0.5 * dist_sq)

    def _fit_gp(self):
        if len(self.X) < 2:
            return False
        X = np.array(self.X)
        Y = np.array(self.Y)
        n = len(X)
        
        K = self._rbf_kernel(X, X)
        K += 1e-6 * np.eye(n)
        
        try:
            self.L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, Y))
            return True
        except np.linalg.LinAlgError:
            K += 1e-4 * np.eye(n)
            try:
                self.L = np.linalg.cholesky(K)
                self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, Y))
                return True
            except np.linalg.LinAlgError:
                return False

    def _predict(self, X_new):
        if self.L is None:
            return None, None
        X_new = np.asarray(X_new)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
            
        K_s = self._rbf_kernel(X_new, np.array(self.X))
        mu = K_s.dot(self.alpha)
        v = np.linalg.solve(self.L, K_s.T)
        var = self.sigma**2 - np.sum(v**2, axis=0)
        var = np.maximum(var, 1e-8)
        std = np.sqrt(var)
        return mu, std

    def _expected_improvement(self, mu, std, f_min):
        if std < 1e-8:
            return 0.0
        Z = (f_min - mu) / std
        return (f_min - mu) * norm.cdf(Z) + std * norm.pdf(Z)

    def __call__(self, func):
        self.start_time = time.time()
        self.func = func
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = None
        self.X = []
        self.Y = []
        
        d = self.dim
        
        # 1. Initial Sampling
        n_init = min(10, self.budget)
        for _ in range(n_init):
            if self._check_time():
                break
            x = np.random.uniform(-5.0, 5.0, size=d)
            f = self._evaluate(x)
            if f is None:
                continue
            self.X.append(x)
            self.Y.append(f)
            
        if self.best_x is None:
             return np.zeros(d), float('inf')

        # 2. Bayesian Optimization Loop
        while self.calls < self.budget and not self._check_time():
            if not self._fit_gp():
                # Fallback to random
                x = np.random.uniform(-5.0, 5.0, size=d)
                f = self._evaluate(x)
                if f is not None:
                    self.X.append(x)
                    self.Y.append(f)
                continue
                
            n_candidates = 200
            # Global
            X_cand_global = np.random.uniform(-5.0, 5.0, size=(n_candidates//2, d))
            # Local
            X_cand_local = np.random.normal(self.best_x, 1.0, size=(n_candidates//2, d))
            X_cand_local = np.clip(X_cand_local, -5.0, 5.0)
            
            X_cand = np.vstack([X_cand_global, X_cand_local])
            
            mu, std = self._predict(X_cand)
            if mu is None:
                continue
                
            ei = np.array([self._expected_improvement(mu[i], std[i], self.best_f) for i in range(len(mu))])
            idx = np.argmax(ei)
            x_next = X_cand[idx]
            
            f_next = self._evaluate(x_next)
            if f_next is None:
                break
                
            self.X.append(x_next)
            self.Y.append(f_next)

        return self.best_x, self.best_f
