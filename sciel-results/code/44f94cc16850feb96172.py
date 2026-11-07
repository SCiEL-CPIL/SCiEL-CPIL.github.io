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
        
        # Population-based CMA-ES
        best_x, best_f, total_calls = self._run_population_cma(func, deadline)
        
        # Final local polish with (1+1)-ES using remaining budget
        remaining = self.budget - total_calls
        if remaining > 5 and time.time() < deadline:
            best_x, best_f, extra_calls = self._run_es(func, deadline, best_x, best_f, remaining)
            total_calls += extra_calls
        
        return best_x, best_f

    def _run_population_cma(self, func, deadline):
        """
        Population-based CMA-ES with covariance adaptation.
        Maintains a population of λ candidates, ranks them,
        and updates mean, step size, and covariance matrix.
        """
        dim = self.dim
        budget = self.budget
        bound = self.bound
        
        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))  # ~13 for dim=5
        mu = lam // 2  # number of best individuals used for update
        
        # Initial parameters
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)  # covariance matrix
        pc = np.zeros(dim)  # evolution path for covariance
        ps = np.zeros(dim)  # evolution path for step size
        
        # CMA-ES parameters
        cmu = 1.0 / (4 + dim / 2)
        cs = (mu + 2) / (dim + mu + 5)
        c1 = (2 - cmu + 2 * mu / dim) / ((dim + 2) ** 2 + 2)
        cmu = 1.0 / (4 + dim / 2)
        damps = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1) - 1) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Cholesky factor of C
        C_sqrt_inv = np.linalg.cholesky(C)
        
        rng = np.random.default_rng(42)
        total_calls = 0
        best_x = x_mean.copy()
        best_f = float('inf')
        
        # Initial population to get baseline
        # Evaluate initial point
        best_f = func(x_mean)
        total_calls += 1
        best_x = x_mean.copy()
        
        # Track best for early stopping
        no_improve_count = 0
        patience = 50  # restart if no improvement for this many generations
        
        while total_calls < budget and time.time() < deadline:
            # Check if we should restart
            if no_improve_count >= patience:
                # Restart with new random center
                x_mean = rng.uniform(-bound * 0.8, bound * 0.8, dim)
                sigma = 2.0
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                C_sqrt_inv = np.linalg.cholesky(C)
                no_improve_count = 0
                # Evaluate new center
                f_new = func(x_mean)
                total_calls += 1
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_mean.copy()
                continue
            
            # Generate population
            if total_calls + lam > budget or time.time() > deadline - 1.0:
                break
            
            # Compute Cholesky of C
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize C
                C += 1e-10 * np.eye(dim)
                C_sqrt = np.linalg.cholesky(C)
            
            # Sample population
            pop_x = []
            pop_f = []
            for i in range(lam):
                z = rng.standard_normal(dim)
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
            pop_f = [pop_f[i] for i in sorted_idx]
            
            # Update mean
            old_mean = x_mean.copy()
            for i in range(min(mu, len(pop_x))):
                x_mean += weights[i] * (pop_x[i] - x_mean)
            
            # Update step size (CSA - Cumulative Step Adaptation)
            # Project new mean onto old mean direction
            if np.linalg.norm(old_mean - x_mean) > 1e-16:
                hsig = 1 if np.linalg.norm(old_mean - x_mean) / (sigma * np.linalg.norm(C_sqrt @ np.ones(dim))) < (1.4 + 2/(dim+1)) * chi_n else 0
                ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mueff) * (C_sqrt_inv @ (x_mean - old_mean)) / sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 1e3)
            else:
                ps = (1 - cs) * ps
            
            # Update covariance matrix
            y_diff = np.array([pop_x[i] - old_mean for i in range(min(mu, len(pop_x)))])
            # Rank-mu update
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

    def _run_es(self, func, deadline, start_x=None, start_f=None, max_calls=None):
        """(1+1)-ES for final polishing."""
        if max_calls is None:
            max_calls = self.budget
        rng = np.random.default_rng(123)
        
        if start_x is not None:
            x = start_x.copy()
            fx = start_f
            calls = 0
        else:
            x = rng.uniform(-self.bound, self.bound, self.dim)
            fx = func(x)
            calls = 1
        
        sigma = 0.5  # smaller sigma for polishing
        
        for _ in range(max_calls - calls):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -self.bound, self.bound)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.85
            if sigma < 1e-10:
                break
        
        return x, fx, calls
