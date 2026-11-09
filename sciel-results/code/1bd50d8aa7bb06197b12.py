"""Optimized solver for BBOB f9 (rotated Rosenbrock) using CMA-ES + L-BFGS-B + BFGS polish."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES global search + gradient-based polish for rotated Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # State tracking
        call_count = [0]
        best_f = [np.inf]
        best_x = [None]

        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return best_f[0]
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x_clipped.copy()
            return f

        def remaining_time():
            return deadline - time.time()

        def has_budget():
            return call_count[0] < budget and remaining_time() > 3.0

        bounds = [(-5.0, 5.0)] * dim
        rng = np.random.default_rng(123)

        # Phase 1: CMA-ES global search to locate the rotated valley
        # CMA-ES adapts its covariance matrix to the landscape anisotropy
        # which is ideal for the curved, narrow valley of Rosenbrock
        cmaes_budget = min(250, budget - 100)  # leave room for polish
        cmaes_max_calls = cmaes_budget

        # Initialize CMA-ES manually to control budget
        # Standard CMA-ES with population size ~4*dim
        pop_size = max(4 * dim, 15)
        sigma = 2.0  # initial step size
        x0 = np.zeros(dim)
        
        # CMA-ES state
        mean = x0.copy()
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_chol = np.eye(dim)
        C_inv_chol = np.eye(dim)
        sigma = 2.0
        step_size = sigma
        mu = pop_size // 2  # number of recombination weights
        weights = np.zeros(mu)
        # Logarithmic weights
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State vectors
        ps = np.zeros(dim)
        hsig = 0.0
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Track best
        best_f_cmaes = np.inf
        best_x_cmaes = np.zeros(dim)
        
        cmaes_calls = 0
        max_cmaes_iter = cmaes_budget // pop_size + 2
        
        for gen in range(max_cmaes_iter):
            if not has_budget() or cmaes_calls >= cmaes_max_calls:
                break
            if remaining_time() < 30.0:
                break
            
            # Generate population
            pop = []
            for i in range(pop_size):
                # Sample from N(mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                x_i = mean + sigma * C_chol @ z
                pop.append(x_i)
            
            # Evaluate population
            pop_f = []
            for x_i in pop:
                if not has_budget():
                    break
                f_i = counted_func(x_i)
                cmaes_calls += 1
                pop_f.append(f_i)
            
            if len(pop_f) < pop_size:
                break
            
            # Sort by fitness
            indices = np.argsort(pop_f)
            pop = [pop[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]
            
            # Update best
            if pop_f[0] < best_f_cmaes:
                best_f_cmaes = pop_f[0]
                best_x_cmaes = pop[0].copy()
            
            # Recombination
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * (pop[i] - mean) / sigma
            
            # Path updates
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * C_inv_chol @ y_w
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**((gen + 1) / (cs * dim))) <= (1.4 + 2 / (dim + 1)) * chi_n
            
            # Covariance matrix update
            delta_hsig = (1 - hsig) * cs * (2 - cs)
            C = (1 - c1 - cmu + delta_hsig * c1) * C + \
                c1 * (np.outer(ps, ps) + (1 - hsig) * (2 - cs) * C) + \
                cmu * np.sum(weights[i] * np.outer(y_w + cs * ps - y_w, y_w + cs * ps - y_w) for i in range(mu))
            
            # Update Cholesky decomposition
            try:
                C_chol = np.linalg.cholesky(C)
                C_inv_chol = np.linalg.inv(C_chol)
            except np.linalg.LinAlgError:
                # Regularize and retry
                C += 1e-10 * np.eye(dim)
                try:
                    C_chol = np.linalg.cholesky(C)
                    C_inv_chol = np.linalg.inv(C_chol)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_chol = np.eye(dim)
                    C_inv_chol = np.eye(dim)
            
            # Update mean
            mean = mean + sigma * y_w
            
            # Adapt step size
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Clip mean to bounds
            mean = np.clip(mean, -5.0, 5.0)
            
            # Stop if step size is too small
            if sigma < 1e-10:
                break
        
        # Use CMA-ES best as starting point for polish
        overall_best_f = best_f_cmaes if np.isfinite(best_f_cmaes) else best_f[0]
        overall_best_x = best_x_cmaes.copy() if np.isfinite(best_f_cmaes) else (best_x[0].copy() if best_x[0] is not None else np.zeros(dim))
        
        # Also consider the tracked best from counted_func
        if best_x[0] is not None and best_f[0] < overall_best_f:
            overall_best_f = best_f[0]
            overall_best_x = best_x[0].copy()

        # Phase 2: L-BFGS-B polish from CMA-ES best
        if has_budget() and remaining_time() > 20.0:
            try:
                res = minimize(
                    counted_func,
                    overall_best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                if res.fun < overall_best_f:
                    overall_best_f = res.fun
                    overall_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 3: BFGS polish
        if has_budget() and remaining_time() > 15.0:
            try:
                res = minimize(
                    counted_func,
                    overall_best_x,
                    method='BFGS',
                    bounds=bounds,
                    options={'maxiter': 100, 'gtol': 1e-14, 'disp': False}
                )
                if res.fun < overall_best_f:
                    overall_best_f = res.fun
                    overall_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 4: Tight L-BFGS-B polish
        if has_budget() and remaining_time() > 10.0:
            try:
                res = minimize(
                    counted_func,
                    overall_best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
                )
                if res.fun < overall_best_f:
                    overall_best_f = res.fun
                    overall_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 5: Final ultra-tight BFGS polish
        if has_budget() and remaining_time() > 5.0:
            try:
                res = minimize(
                    counted_func,
                    overall_best_x,
                    method='BFGS',
                    bounds=bounds,
                    options={'maxiter': 50, 'gtol': 1e-16, 'disp': False}
                )
                if res.fun < overall_best_f:
                    overall_best_f = res.fun
                    overall_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Use tracked best if better
        if best_x[0] is not None and best_f[0] < overall_best_f:
            overall_best_f = best_f[0]
            overall_best_x = best_x[0]

        if overall_best_x is None:
            if best_x[0] is not None:
                return best_x[0], best_f[0]
            return np.zeros(dim), 0.0

        return overall_best_x, overall_best_f
