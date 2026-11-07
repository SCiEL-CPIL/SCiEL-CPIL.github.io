import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin
        
        # Phase 1: Global search with CMA-ES (use ~70% of budget)
        cma_budget = int(0.7 * self.budget)
        best_x, best_f, total_calls = self._run_cma(func, deadline, cma_budget)
        
        # Phase 2: Local refinement with (1+1)-ES using remaining budget
        remaining = self.budget - total_calls
        if remaining > 10 and time.time() < deadline:
            best_x, best_f, extra_calls = self._run_polish_es(func, deadline, best_x, best_f, remaining)
            total_calls += extra_calls
        
        return best_x, best_f

    def _run_cma(self, func, deadline, budget):
        """
        Correct CMA-ES implementation with proper covariance adaptation.
        """
        dim = self.dim
        bound = self.bound
        
        # Population size - larger for better exploration
        lam = max(4, int(4 + 3 * np.log(dim)))  # ~13 for dim=5
        mu = lam // 2
        
        # Initial parameters
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        ps = np.zeros(dim)
        
        # CMA-ES parameters (standard values)
        mueff = 1.0 / (2.0 / np.sum(np.log(np.arange(1, mu + 1)) - np.log(np.arange(mu, 0, -1))))
        c1 = (2 - mueff / (dim + 1) + 2 * mueff / dim) / ((dim + 1.3) ** 2 + mueff)
        cmu = (mueff - 2 + 2 / mueff) / ((dim + 2) ** 2 + mueff)
        cs = (mueff + 2) / (dim + mueff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        rng = np.random.default_rng(42)
        total_calls = 0
        best_x = x_mean.copy()
        best_f = float('inf')
        
        # Evaluate initial point
        best_f = func(x_mean)
        total_calls += 1
        best_x = x_mean.copy()
        
        no_improve_count = 0
        patience = 40
        
        while total_calls < budget and time.time() < deadline:
            # Restart if stuck
            if no_improve_count >= patience:
                x_mean = rng.uniform(-bound * 0.8, bound * 0.8, dim)
                sigma = 2.0
                C = np.eye(dim)
                ps = np.zeros(dim)
                no_improve_count = 0
                f_new = func(x_mean)
                total_calls += 1
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_mean.copy()
                continue
            
            if total_calls + lam > budget or time.time() > deadline - 1.0:
                break
            
            # Sample population
            pop_x = []
            pop_f = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                # Use Cholesky decomposition for sampling from C
                try:
                    C_sqrt = np.linalg.cholesky(C)
                    y = x_mean + sigma * (C_sqrt @ z)
                except np.linalg.LinAlgError:
                    C += 1e-10 * np.eye(dim)
                    C_sqrt = np.linalg.cholesky(C)
                    y = x_mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -bound, bound)
                pop_x.append(y)
                if total_calls + 1 > budget or time.time() > deadline - 0.5:
                    break
                f_val = func(y)
                pop_f.append(f_val)
                total_calls += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = y.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            if len(pop_x) < 2:
                break
            
            # Sort by fitness
            sorted_idx = np.argsort(pop_f)
            pop_x = [pop_x[i] for i in sorted_idx]
            
            # Update mean
            old_mean = x_mean.copy()
            for i in range(min(mu, len(pop_x))):
                x_mean += weights[i] * (pop_x[i] - x_mean)
            
            # Update step size
            if np.linalg.norm(old_mean - x_mean) > 1e-16:
                try:
                    C_sqrt = np.linalg.cholesky(C)
                    hsig = 1 if np.linalg.norm(old_mean - x_mean) / (sigma * np.linalg.norm(C_sqrt @ np.ones(dim))) < (1.4 + 2/(dim+1)) * chi_n else 0
                    ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mueff) * (np.linalg.solve(C, (x_mean - old_mean) / sigma))
                    sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                    sigma = np.clip(sigma, 1e-12, 1e3)
                except:
                    pass
            else:
                ps = (1 - cs) * ps
            
            # Update covariance matrix
            y_diff = np.array([pop_x[i] - old_mean for i in range(min(mu, len(pop_x)))])
            C = (1 - c1 - cmu) * C
            for i in range(min(mu, len(pop_x))):
                y_vec = y_diff[i] / sigma
                C += c1 * np.outer(y_vec, y_vec)
                C += cmu * weights[i] * np.outer(y_vec, y_vec)
            
            # Symmetrize and regularize
            C = 0.5 * (C + C.T)
            C += 1e-12 * np.eye(dim)
            
            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-10:
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = 0.5 * (C + C.T)
        
        return best_x, best_f, total_calls

    def _run_polish_es(self, func, deadline, start_x, start_f, max_calls):
        """(1+1)-ES for final polishing with adaptive step size."""
        rng = np.random.default_rng(123)
        x = start_x.copy()
        fx = start_f
        calls = 0
        sigma = 0.3  # start with smaller sigma for local refinement
        
        for _ in range(max_calls):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -self.bound, self.bound)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.9
            if sigma < 1e-8:
                break
        
        return x, fx, calls
