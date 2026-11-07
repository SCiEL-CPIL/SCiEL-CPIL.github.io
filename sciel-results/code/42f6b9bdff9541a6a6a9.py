"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES style optimizer for BBOB f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(123)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def remaining():
            return self.budget - calls_used
        
        def time_ok():
            return time.time() < deadline
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        mu = max(1, dim // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 - mu_eff / (4 * (dim + 1.3)) + c_c) / ((dim + 1.5 + c_c) * (2 - mu_eff / (4 * (dim + 1.3)) + c_c + 1))
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((dim + 2) * (mu_eff + 2 - 2.0 / mu_eff)))
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))
        
        def cma_es(x0, sigma0, max_calls):
            nonlocal best_x, best_f, calls_used
            
            x = x0.copy()
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            damps = 1.0 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            while time_ok() and remaining() > 0:
                # Generate population
                offspring = []
                for i in range(mu):
                    z = rng.standard_normal(dim)
                    y = x + sigma * (B @ (D * z))
                    offspring.append(y)
                
                # Evaluate offspring
                fvals = []
                for y in offspring:
                    y_clipped = np.clip(y, -5, 5)
                    fy = func(y_clipped)
                    calls_used += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y_clipped.copy()
                    fvals.append(fy)
                
                # Sort by fitness
                idx = np.argsort(fvals)
                offspring = [offspring[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Update mean
                x_old = x.copy()
                for i in range(mu):
                    x += w[i] * (offspring[i] - x_old)
                
                # Update step size
                y_mean = (x - x_old) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * y_mean
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Update covariance matrix
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 1 / dim) * chi_n:
                    v = y_mean
                else:
                    v = ps
                pc = (1 - c_c) * pc + c_c * v
                C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - (1 - c_c)**2) * np.eye(dim))
                for i in range(mu):
                    zi = (offspring[i] - x_old) / sigma
                    C += c_mu * w[i] * np.outer(zi, zi)
                
                # Eigen-decompose for stable sampling
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    D = np.sqrt(eigvals)
                    B = eigvecs
                except np.linalg.LinAlgError:
                    break
                
                # Check convergence
                if sigma < 1e-8:
                    break
                
                # If we've used too many calls, break
                if remaining() <= 0:
                    break
            
            return x, fx
        
        # Multi-start strategy with CMA-ES
        # First, do a coarse grid to find promising regions
        n_starts = 8
        start_budget = max(10, budget // (n_starts * 3))
        
        # Generate diverse starting points
        starts = []
        # Origin and axis-aligned points
        starts.append(np.zeros(dim))
        for i in range(dim):
            p = np.zeros(dim)
            p[i] = 2.0
            starts.append(p)
        # Random points
        for _ in range(max(0, n_starts - 1 - dim)):
            starts.append(rng.uniform(-4, 4, dim))
        
        # Evaluate starting points
        start_fvals = []
        for s in starts:
            if not time_ok() or remaining() <= 0:
                break
            s_clipped = np.clip(s, -5, 5)
            fs = func(s_clipped)
            calls_used += 1
            if fs < best_f:
                best_f = fs
                best_x = s_clipped.copy()
            start_fvals.append(fs)
        
        # Sort starts by fitness
        if start_fvals:
            order = np.argsort(start_fvals)
            starts_sorted = [starts[i] for i in order]
            fvals_sorted = [start_fvals[i] for i in order]
        else:
            starts_sorted = starts
            fvals_sorted = start_fvals
        
        # Run CMA-ES from best starting points
        n_cma = min(len(starts_sorted), 4)
        for i in range(n_cma):
            if not time_ok() or remaining() <= 0:
                break
            # Allocate remaining budget to CMA-ES runs
            alloc = remaining() // (n_cma - i)
            if alloc < 20:
                alloc = remaining()
            cma_es(starts_sorted[i].copy(), 1.0, alloc)
        
        # If we still have budget, do some local refinement with (1+1)-ES
        if time_ok() and remaining() > 10:
            x = best_x.copy() if best_x is not None else rng.uniform(-4, 4, dim)
            fx = best_f
            sigma = 0.5
            while time_ok() and remaining() > 0:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.3, 5.0)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.8, 0.01)
                    if sigma <= 0.011:
                        break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
