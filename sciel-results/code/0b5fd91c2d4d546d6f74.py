"""Rewritten program using MODEL-AND-OPTIMIZE with CMA-ES polish for Sphere."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Model-based optimizer for BBOB-style Sphere functions with CMA-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(self.dim)
        
        bounds = [(-5.0, 5.0)] * self.dim
        N = self.dim

        def wrapped_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            
            x = np.clip(x, -5.0, 5.0)
            
            val = func(x)
            self.calls += 1
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        # PHASE 1: Estimate the center x* of the Sphere
        # f(x) = ||x - x*||^2 + c
        # f(x) = x^T x - 2 x*^T x + (||x*||^2 + c)
        # Let y = f(x) - x^T x. Then y = -2 x*^T x + K.
        # We use N+1 points: Origin and N axis points.
        
        D = 4.0
        points = [np.zeros(N)]
        for i in range(N):
            p = np.zeros(N)
            p[i] = D
            points.append(p)
        
        values = []
        for p in points:
            try:
                v = wrapped_func(p)
                values.append(v)
            except (ValueError, TimeoutError, Exception):
                break
        
        if len(values) < N + 1:
            # Fallback: Random search + CMA-ES if model fitting fails
            rng = np.random.default_rng(42)
            for _ in range(10):
                if self.calls >= self.budget: break
                x = rng.uniform(-5, 5, N)
                wrapped_func(x)
            self._cma_es_polish(wrapped_func, self.best_x, deadline)
            return (self.best_x, self.best_f)

        # Solve for x* and K
        # y_i = -2 x_i^T x* + K
        # A x* = b, where A is N x N, b is N x 1
        # Using x_0 as reference:
        # y_i - y_0 = -2 (x_i - x_0)^T x*
        # (y_0 - y_i) / 2 = (x_i - x_0)^T x*
        
        A = np.zeros((N, N))
        b = np.zeros(N)
        
        x0 = points[0]
        y0 = values[0] - np.dot(x0, x0)
        
        for i in range(1, N + 1):
            xi = points[i]
            yi = values[i] - np.dot(xi, xi)
            
            diff = xi - x0
            A[i-1] = diff
            b[i-1] = (y0 - yi) / 2.0
        
        try:
            x_est = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x_est = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Clip estimated center to bounds
        x_start = np.clip(x_est, -5.0, 5.0)
        
        # Evaluate start point
        try:
            v = wrapped_func(x_start)
        except (ValueError, TimeoutError, Exception):
            pass

        # PHASE 2: CMA-ES polish
        # CMA-ES adapts the covariance matrix to handle rotation and
        # automatically adjusts step sizes based on function curvature.
        self._cma_es_polish(wrapped_func, x_start, deadline)
        
        return (self.best_x, self.best_f)

    def _cma_es_polish(self, wrapped_func, x0, deadline):
        """CMA-ES implementation for polishing the solution."""
        N = self.dim
        rng = np.random.default_rng(123)
        
        # CMA-ES parameters
        # Population size
        lambda_pop = max(4 + int(3 * np.log(N)), 10)
        
        # Initial step size
        sigma = 0.5  # Start with moderate step size
        
        # Mean
        m = x0.copy()
        
        # Covariance matrix (start isotropic)
        C = np.eye(N)
        
        # Eigen-decomposition cache
        D = np.sqrt(np.diag(C))
        B = np.eye(N)
        
        # Adaptation parameters
        cs = (lambda_pop + 2) / (N + lambda_pop + 5)
        ds = 1 + 2 * max(0, (N - 1) / (N + lambda_pop) - 1) + cs
        cmu = min(1 - cs, 2 / ((N + 1)**2 + lambda_pop))
        cmu = max(0.1, cmu)
        
        # Step-size adaptation
        chi_n = np.sqrt(N) * (1 - 1/(4*N) + 1/(21*N**2))
        
        # Path accumulators
        p_sigma = np.zeros(N)
        p_c = np.zeros(N)
        
        # Max iterations
        max_iter = 200
        
        for it in range(max_iter):
            if self.calls >= self.budget or time.time() > deadline:
                break
            
            # Update eigen-decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)  # Ensure positive
            D = np.sqrt(eigvals)
            B = eigvecs
            
            # Sample population
            offspring = []
            for i in range(lambda_pop):
                z = rng.standard_normal(N)
                x_i = m + sigma * (B @ (D * z))
                offspring.append(x_i)
            
            # Evaluate offspring
            fit_vals = []
            for x_i in offspring:
                try:
                    f_i = wrapped_func(x_i)
                    fit_vals.append(f_i)
                except (ValueError, TimeoutError, Exception):
                    fit_vals.append(float('inf'))
            
            # Sort by fitness
            sort_idx = np.argsort(fit_vals)
            offspring_sorted = [offspring[i] for i in sort_idx]
            fit_vals_sorted = [fit_vals[i] for i in sort_idx]
            
            # Update mean
            w = np.exp(-np.arange(1, lambda_pop + 1) / (0.5 * min(lambda_pop / 2, np.log(lambda_pop))))
            w /= np.sum(w)
            w = w[:lambda_pop]
            
            # Effective population size
            mu_eff = 1.0 / np.sum(w**2)
            
            # Update mean
            m_old = m.copy()
            m = np.sum([w[i] * offspring_sorted[i] for i in range(lambda_pop)], axis=0)
            
            # Update step size (CSA)
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (B @ (D @ ((m - m_old) / (sigma * D))))
            # Simplified: p_sigma = (1-cs)*p_sigma + sqrt(cs*(2-cs)*mu_eff) * (B @ ((m - m_old) / sigma))
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (B @ ((m - m_old) / sigma))
            
            # Hoeffding adaptation
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**(2 * (it + 1))) < (1.4 + 2 / (N + 1)) * chi_n else 0
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = min(sigma, 1e6)  # Cap step size
            
            # Update covariance matrix
            p_c = (1 - cmu) * p_c + h_sigma * np.sqrt(cmu * (2 - cmu) * mu_eff) * (B @ ((m - m_old) / sigma))
            
            C = (1 - cmu) * C + cmu * (np.outer(p_c, p_c) + 1) + (1 - cmu) * np.sum(
                [w[i] * np.outer(B @ (D * (B.T @ (offspring_sorted[i] - m_old) / sigma)), 
                               B @ (D * (B.T @ (offspring_sorted[i] - m_old) / sigma))) 
                 for i in range(lambda_pop)], axis=0
            )
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Check convergence
            if np.linalg.norm(m - m_old) < 1e-10 * (1 + np.linalg.norm(m)):
                break
            
            # Early stopping if we've converged well
            if fit_vals_sorted[0] < 1e-6:
                break
        
        return
# EVOLVE-BLOCK-END
