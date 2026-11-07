import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 290.0

    def __call__(self, func):
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 290.0
        
        def wrapped_func(x):
            self.call_count += 1
            if time.time() > self.deadline:
                raise TimeoutError
            if self.call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            # Clip to bounds
            x = np.clip(x, -self.bound, self.bound)
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Phase 1: Global Search with Random Sampling
        # Use more points to increase probability of finding the global basin
        # 100 points is a reasonable trade-off for 5D
        n_global_samples = 100
        for _ in range(n_global_samples):
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            p = np.random.uniform(-self.bound, self.bound, self.dim)
            wrapped_func(p)
            
        if self.best_x is None:
             # Fallback if no calls made? Should not happen
             return np.zeros(self.dim), 0.0

        # Phase 2: CMA-ES for rotation-aware local optimization
        # CMA-ES adapts its covariance matrix to the local landscape geometry,
        # which handles the hidden rotation in f21 better than Powell or Nelder-Mead
        try:
            self._run_cmaes(wrapped_func, self.best_x, budget_limit=300)
        except Exception:
            pass

        # Phase 3: High-precision polish with CMA-ES if budget remains
        if self.call_count < self.budget - 50 and time.time() < self.deadline - 10:
            try:
                self._run_cmaes_polish(wrapped_func, self.best_x, budget_limit=150)
            except Exception:
                pass

        return self.best_x, self.best_f

    def _run_cmaes(self, wrapped_func, x0, budget_limit):
        """Run CMA-ES optimization with adaptive population size."""
        dim = self.dim
        bound = self.bound
        
        # Initial parameters
        popsize = min(4 + int(3 * np.log(dim)), 10)  # Standard CMA-ES popsize
        sigma = 1.0  # Initial step size
        max_iter = 100
        
        # CMA-ES state
        x_mean = np.array(x0, dtype=float)
        C = np.eye(dim)  # Covariance matrix
        step_sizes = np.ones(dim)  # Eigenvalues of C (for square root)
        
        # Strategies
        c_sigma = (popsize + 2) / (dim + popsize + 5)
        c_c = (4 + dim / popsize) / (dim + 4 + 2 * popsize / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + popsize)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2) / ((dim + 2) ** 2 + popsize))
        mu_eff = popsize / 2  # Effective population size
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        d_corr = 1 + 2 * max(0, (c_c + 1) / (c_sigma + 1) - 1)
        
        for iteration in range(max_iter):
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            if self.call_count + popsize > self.budget:
                popsize = self.budget - self.call_count
                if popsize < 4:
                    break
            
            # Generate population
            B = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            D = np.diag(np.sqrt(np.diag(C)))
            
            points = []
            for i in range(popsize):
                z = np.random.randn(dim)
                x_new = x_mean + sigma * B @ z
                x_new = np.clip(x_new, -bound, bound)
                points.append(x_new)
            
            # Evaluate population
            values = []
            for x in points:
                if self.call_count >= self.budget or time.time() > self.deadline:
                    break
                values.append(wrapped_func(x))
            
            if len(values) < popsize:
                break
            
            # Sort by fitness
            order = np.argsort(values)
            points = np.array([points[i] for i in order])
            values = np.array([values[i] for i in order])
            
            # Weights for mean update
            weights = np.log(popsize / 2 + 1) - np.log(np.arange(1, popsize + 1))
            weights /= np.sum(weights)
            mu = min(mu_eff, popsize // 2)
            weights = weights[:mu]
            weights /= np.sum(weights)
            
            # Update mean
            x_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * points[:mu], axis=0)
            
            # Update sigma
            if np.linalg.norm(x_mean - x_old) > 1e-16:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x_mean - x_old) / (sigma * np.linalg.norm(B))
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update C
            if np.linalg.norm(x_mean - x_old) > 1e-16:
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_old) / sigma
            
            delta_h = 1 if h_sigma else 0
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * np.sum(weights[:, None, None] * np.outer(points[:mu] - x_old, points[:mu] - x_old), axis=0)
            
            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update best
            if values[0] < self.best_f:
                self.best_f = values[0]
                self.best_x = points[0].copy()

    def _run_cmaes_polish(self, wrapped_func, x0, budget_limit):
        """Run high-precision CMA-ES polish with smaller step size."""
        dim = self.dim
        bound = self.bound
        
        # Smaller initial sigma for polishing
        sigma = 0.01
        popsize = min(4 + int(3 * np.log(dim)), 6)
        max_iter = 50
        
        x_mean = np.array(x0, dtype=float)
        C = np.eye(dim)
        
        c_sigma = (popsize + 2) / (dim + popsize + 5)
        c_c = (4 + dim / popsize) / (dim + 4 + 2 * popsize / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + popsize)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2) / ((dim + 2) ** 2 + popsize))
        mu_eff = popsize / 2
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        for iteration in range(max_iter):
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            if self.call_count + popsize > self.budget:
                popsize = self.budget - self.call_count
                if popsize < 4:
                    break
            
            B = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            
            points = []
            for i in range(popsize):
                z = np.random.randn(dim)
                x_new = x_mean + sigma * B @ z
                x_new = np.clip(x_new, -bound, bound)
                points.append(x_new)
            
            values = []
            for x in points:
                if self.call_count >= self.budget or time.time() > self.deadline:
                    break
                values.append(wrapped_func(x))
            
            if len(values) < popsize:
                break
            
            order = np.argsort(values)
            points = np.array([points[i] for i in order])
            values = np.array([values[i] for i in order])
            
            weights = np.log(popsize / 2 + 1) - np.log(np.arange(1, popsize + 1))
            weights /= np.sum(weights)
            mu = min(mu_eff, popsize // 2)
            weights = weights[:mu]
            weights /= np.sum(weights)
            
            x_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * points[:mu], axis=0)
            
            if np.linalg.norm(x_mean - x_old) > 1e-16:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x_mean - x_old) / (sigma * np.linalg.norm(B))
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            if np.linalg.norm(x_mean - x_old) > 1e-16:
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_old) / sigma
            
            delta_h = 1 if h_sigma else 0
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * np.sum(weights[:, None, None] * np.outer(points[:mu] - x_old, points[:mu] - x_old), axis=0)
            
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            if values[0] < self.best_f:
                self.best_f = values[0]
                self.best_x = points[0].copy()
