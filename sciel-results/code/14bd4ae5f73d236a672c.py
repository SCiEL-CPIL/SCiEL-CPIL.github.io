# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """CMA-ES style optimizer for the Discus (f11) function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.t0 = None
        self.deadline = None

    def _eval(self, func, x):
        x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
        if x.shape != (self.dim,):
            x = x.reshape(self.dim)
        f = float(func(x))
        self.calls += 1
        if not np.isfinite(f):
            f = np.inf
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def _cma_es(self, func, x0, sigma0, budget):
        """Simple CMA-ES implementation."""
        d = self.dim
        # Population size
        lambda_ = 4 + int(3 * np.log(d))
        # Step size
        sigma = sigma0
        x = x0.copy()
        
        # Covariance matrix
        C = np.eye(d)
        # Evolution paths
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Parameters
        c_sigma = (lambda_ + 2) / (d + lambda_ + 5)
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        # Initial population
        pops = []
        fvals = []
        for i in range(lambda_):
            z = np.random.randn(d)
            xi = x + sigma * z
            fval = self._eval(func, xi)
            pops.append(xi)
            fvals.append(fval)
        
        max_iter = (self.budget - self.calls) // lambda_
        for it in range(max_iter):
            if self.calls >= budget:
                break
            if time.time() > self.deadline:
                break
            
            # Sort by fitness
            order = np.argsort(fvals)
            pops = [pops[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Update mean
            x_new = sum(w[i] * pops[i] for i in range(mu))
            
            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (x_new - x)
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (it + 1)))) < (1.4 + 2 / (d + 1)) * chi_n
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            delta_h = (1 - h_sigma) * (1 - (1 - c_c)**(it + 1))
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) - C) + c_mu * sum(w[i] * np.outer(pops[i] - x, pops[i] - x) for i in range(mu))
            
            x = x_new
            
            # Generate new population
            new_pops = []
            new_fvals = []
            for i in range(lambda_):
                if self.calls >= budget:
                    break
                if time.time() > self.deadline:
                    break
                # Cholesky decomposition
                try:
                    L = np.linalg.cholesky(C)
                except:
                    L = np.eye(d)
                z = np.random.randn(d)
                xi = x + sigma * (L @ z)
                fval = self._eval(func, xi)
                new_pops.append(xi)
                new_fvals.append(fval)
            
            pops = new_pops
            fvals = new_fvals
        
        return self.best_x.copy(), self.best_f

    def __call__(self, func):
        self.t0 = time.time()
        self.deadline = self.t0 + 290.0  # safety margin before 330s kill
        d = self.dim
        rng = np.random.default_rng(42)
        
        # Phase 1: Random sampling to get a good initial point
        n_initial = min(50, self.budget // 10)
        best_x = None
        best_f = np.inf
        for i in range(n_initial):
            if self.calls >= self.budget:
                break
            if time.time() > self.deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=d)
            f = self._eval(func, x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(d)
            best_f = self._eval(func, best_x)
        
        # Phase 2: CMA-ES from the best point
        remaining = self.budget - self.calls
        if remaining > 50:
            self._cma_es(func, best_x, 1.0, self.budget)
        
        # Phase 3: Local refinement with Nelder-Mead from the best point
        remaining = self.budget - self.calls
        if remaining > 20:
            x0 = self.best_x.copy()
            res = minimize(func, x0, method='Nelder-Mead', 
                          options={'maxiter': remaining // (d + 1), 'xatol': 1e-8, 'fatol': 1e-10})
            # Count the calls made by Nelder-Mead
            # This is tricky since we don't have direct access, so we'll just use the result
            if np.isfinite(res.fun) and res.fun < self.best_f:
                self.best_f = res.fun
                self.best_x = np.clip(res.x, -5.0, 5.0)
        
        if self.best_x is None:
            self.best_x = np.zeros(d)
            self.best_f = 0.0
        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
