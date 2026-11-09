"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES style optimizer with adaptive step size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Initial population: centered at origin with moderate spread
        x0 = np.zeros(n)
        f0 = func(x0)
        
        # Try a few random starts to find a better initial point
        best_x = x0.copy()
        best_f = f0
        for _ in range(min(20, budget // 10)):
            x_try = rng.uniform(-5, 5, n)
            f_try = func(x_try)
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()
        
        x = best_x.copy()
        fx = best_f
        calls_made = min(20, budget // 10) + 1
        
        # CMA-ES parameters
        lambda_ = max(4, n + 1)  # population size
        mu = lambda_ // 2        # number of parents
        sigma = 1.0              # initial step size
        
        # Weighting
        weights = np.zeros(mu)
        wsum = 0.0
        for i in range(mu):
            weights[i] = np.log(mu + 1) - np.log(i + 1)
            wsum += weights[i]
        weights /= wsum
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        hs = (1 + (1 + 3 / n) * c_sigma) / (1 - c_sigma)
        
        def update_sigma(sigma, p_sigma, hs):
            return sigma * np.exp(c_sigma / hs * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**20) - np.sqrt(1 - np.exp(-2 * mu_eff / n))))
        
        remaining = budget - calls_made
        gen = 0
        no_improve = 0
        
        while remaining > 0 and gen < 1000:
            # Generate population
            Y = np.zeros((lambda_, n))
            for i in range(lambda_):
                Y[i] = x + sigma * rng.standard_normal(n)
                Y[i] = np.clip(Y[i], -5, 5)
            
            # Evaluate
            F = np.zeros(lambda_)
            for i in range(lambda_):
                if remaining <= 0:
                    break
                F[i] = func(Y[i])
                remaining -= 1
            
            if remaining <= 0:
                break
            
            # Sort
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            
            # Update best
            if F[0] < fx:
                fx = F[0]
                x = Y[0].copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # CMA-ES update
            m = np.sum(weights[:, None] * Y[:mu], axis=0)
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * (m - x) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (m - x) / sigma
            
            # Hessian-free rank update
            C = (1 + c_1 * (2 - c_1) - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) - np.eye(n))
            C += c_mu * np.sum(weights[:, None, None] * np.outer(Y[:mu] - x, Y[:mu] - x) for _ in [0]) / mu_eff * mu_eff
            
            # Symmetrize and ensure positive definite
            C = (C + C.T) / 2
            # Add small regularization to ensure PD
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            x = m
            sigma = update_sigma(sigma, p_sigma, hs)
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            gen += 1
            
            # If stuck, try a random restart
            if no_improve > 50 and remaining > 10:
                x_try = rng.uniform(-5, 5, n)
                f_try = func(x_try)
                remaining -= 1
                if f_try < fx:
                    x = x_try
                    fx = f_try
                    sigma = 2.0
                    C = np.eye(n)
                    p_sigma = np.zeros(n)
                    p_c = np.zeros(n)
                    no_improve = 0
        
        return x, fx
# EVOLVE-BLOCK-END
