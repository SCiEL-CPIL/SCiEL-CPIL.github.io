"""Rewritten optimizer for BBOB f22 (Gallagher 21 peaks)."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES based optimizer with local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 280
        rng = np.random.default_rng(42)

        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0

        def bounded_func(x):
            nonlocal calls_made
            x_clipped = np.clip(x, -5.0, 5.0)
            calls_made += 1
            return func(x_clipped)

        # CMA-ES parameters
        N = 10  # Population size
        mu = N // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_w = np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_w + 2) / (self.dim + mu_w + 5)
        c_c = (4 + mu_w / self.dim) / (self.dim + 4 + 2 * mu_w / self.dim)
        c_1 = 2 / ((self.dim + 1.3) ** 2 + mu_w)
        c_mu = min(1 - c_1, (2 * (mu_w - 2 + 1 / mu_w)) / ((self.dim + 2) ** 2 + mu_w))
        
        # State
        m = np.zeros(self.dim)
        sigma = 2.0
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        hsigma = np.sqrt((1 + c_c) * c_c * (N + 2) / (1 - c_c))
        
        # Initial evaluation
        fx0 = bounded_func(m)
        best_f = fx0
        best_x = m.copy()
        
        for gen in range(100): # Max 100 generations * 10 = 1000 calls, but we limit by budget
            if calls_made + N > self.budget:
                break
            if time.time() > deadline - 10:
                break
                
            # Sample population
            pop = []
            for _ in range(N):
                B, D = np.linalg.eigh(C)
                z = rng.standard_normal(self.dim)
                x = m + sigma * (B @ (D * z))
                pop.append(x)
                
            # Evaluate
            fvals = []
            for x in pop:
                fx = bounded_func(x)
                fvals.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = np.clip(x, -5.0, 5.0).copy()
                    
            # Sort
            idx = np.argsort(fvals)
            x_sorted = [pop[i] for i in idx]
            f_sorted = [fvals[i] for i in idx]
            
            # Update mean
            m_new = sum(w[i] * x_sorted[i] for i in range(mu))
            
            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_w) / np.linalg.norm(np.sqrt(C)) * (m_new - m)
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (self.dim + 1)) * hsigma:
                sigma *= np.exp(c_sigma / 5 * (np.linalg.norm(ps) / hsigma - 1))
                
            # Update covariance
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_w) * (m_new - m) / sigma
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (c_c * (2 - c_c) - 1) * C) + \
                c_mu * sum(w[i] * np.outer(x_sorted[i] - m, x_sorted[i] - m) for i in range(mu)) / sigma**2
            
            m = m_new
            
            # Check for convergence
            if np.max(np.abs(np.diag(C))) < 1e-12:
                break

        # Local refinement
        if calls_made < self.budget and time.time() < deadline - 5:
            try:
                res = minimize(
                    bounded_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

        return best_x, best_f
