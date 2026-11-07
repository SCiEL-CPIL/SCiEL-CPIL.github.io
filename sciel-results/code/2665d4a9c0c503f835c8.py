"""Rewritten optimizer using CMA-ES with multi-restart strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-restart for black-box continuous optimization on [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES from x0 with initial step size sigma0 for up to budget evaluations."""
        d = self.dim
        x = x0.copy().astype(np.float64)
        f_x = func(x)
        sigma = sigma0
        C = np.eye(d)
        C_sqrt_inv = np.eye(d)
        
        # CMA-ES parameters
        lambda_ = max(4, 4 + int(3 * np.log(d)))  # population size
        mu = lambda_ // 2  # number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
        
        ps = np.zeros(d)
        pc = np.zeros(d)
        calls = 1
        best_x = x.copy()
        best_f = f_x
        
        for gen in range(budget):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            # Sample population
            Y = np.array([x + sigma * C_sqrt_inv @ rng.standard_normal(d) for _ in range(lambda_)])
            Y = np.clip(Y, -5, 5)
            
            # Evaluate
            F = np.array([func(y) for y in Y])
            calls += len(Y)
            
            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = Y_sorted[0].copy()
            
            # Update mean
            x_old = x.copy()
            x = np.sum(weights[:, np.newaxis] * Y_sorted[:mu], axis=0)
            
            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_sqrt_inv.T @ (x - x_old))
            if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (d + 1)) * (1 - c_c):
                sigma *= np.exp(c_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance
            if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (d + 1)) * (1 - c_c):
                pc = (1 - c_c) * pc + c_c * (x - x_old) / sigma
            delta_sig = (np.linalg.norm(ps) / chi_n - 1) * (1 + c_c)
            C = ((1 - c_1 - c_mu) * C +
                 c_1 * (pc @ pc + (1 - (1 - c_c) ** 2) * np.eye(d)) +
                 c_mu / sigma ** 2 * np.sum(weights[:, np.newaxis, np.newaxis] * 
                                            np.outer(Y_sorted[:mu] - x_old, Y_sorted[:mu] - x_old), axis=0))
            
            # Ensure positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update C_sqrt_inv
            eigvals_sqrt = np.sqrt(np.maximum(eigvals, 1e-12))
            C_sqrt_inv = eigvecs @ np.diag(1 / eigvals_sqrt) @ eigvecs.T
            
            # Termination criteria
            if sigma < 1e-12:
                break
            if np.all(np.abs(x - x_old) < 1e-12):
                break
        
        return best_x, best_f, calls

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        total_budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 280  # 280s out of 300s limit
        
        # Multi-restart strategy
        # First restart: center, large sigma
        # Subsequent restarts: around best found, smaller sigma
        
        restarts = [
            (np.zeros(d), 2.0, 200),   # center, sigma=2, 200 evals
            (np.zeros(d), 1.0, 100),   # center, sigma=1, 100 evals
            (np.zeros(d), 0.5, 50),    # center, sigma=0.5, 50 evals
            (np.zeros(d), 0.2, 30),    # center, sigma=0.2, 30 evals
            (np.zeros(d), 0.1, 20),    # center, sigma=0.1, 20 evals
        ]
        
        # Adjust restarts based on budget
        if total_budget <= 100:
            restarts = [(np.zeros(d), 2.0, total_budget)]
        elif total_budget <= 200:
            restarts = [
                (np.zeros(d), 2.0, 120),
                (np.zeros(d), 1.0, 60),
                (np.zeros(d), 0.5, 20),
            ]
        elif total_budget <= 300:
            restarts = [
                (np.zeros(d), 2.0, 150),
                (np.zeros(d), 1.0, 80),
                (np.zeros(d), 0.5, 40),
                (np.zeros(d), 0.2, 30),
            ]
        
        best_x = None
        best_f = np.inf
        used_calls = 0
        
        for x0, sigma0, budget_i in restarts:
            if time.time() > deadline:
                break
            if used_calls >= total_budget:
                break
            
            remaining = total_budget - used_calls
            budget_i = min(budget_i, remaining)
            
            if budget_i < 10:
                break
            
            # If we have a previous best, start near it
            if best_x is not None and best_f < np.inf:
                # Perturb around best
                x0 = np.clip(best_x + 0.5 * sigma0 * rng.standard_normal(d), -5, 5)
            
            x_best, f_best, calls_i = self._cma_es(func, x0, sigma0, budget_i, rng, deadline)
            used_calls += calls_i
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
        
        # If no restarts were executed (shouldn't happen), do a simple search
        if best_x is None:
            x = np.zeros(d)
            best_f = func(x)
            best_x = x.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
