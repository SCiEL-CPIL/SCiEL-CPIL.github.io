"""CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population, adaptation, and time-safe budgeting."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        remaining = budget

        # CMA-ES parameters
        # lambda: population size
        lam = int(4 + 3 * np.log(dim))
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1 / np.sum(w**2)
        
        # Initial point and sigma
        rng = np.random.default_rng(42)
        x = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        # Path accumulators
        ps = np.zeros(dim)
        phis = np.zeros(dim)
        
        # Adaptation parameters
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(20*dim**2))
        c_sigma = (1 + 2/min(1/(lam+1), (dim-1)/(dim+3))) / (dim + 8)
        c_c = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        c_1 = 2 / ((dim+1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1/mueff) / ((dim+2)**2 + mueff))
        
        # Check termination
        def check_termination():
            # Simple stagnation check
            return False

        # Main loop
        while remaining > 0 and time.time() < deadline - 2.0:
            # Generate population
            pops = []
            fitvals = []
            
            # Compute sqrt of C for sampling
            # Use Cholesky or eigen decomposition
            try:
                L = np.linalg.cholesky(C)
            except:
                L = np.eye(dim)
            
            for i in range(lam):
                if remaining <= 0 or time.time() > deadline - 2.0:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * L @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                remaining -= 1
                pops.append(y)
                fitvals.append(fy)
            
            if len(pops) == 0:
                break
            
            # Sort by fitness
            sort_idx = np.argsort(fitvals)
            pops = [pops[i] for i in sort_idx]
            fitvals = [fitvals[i] for i in sort_idx]
            
            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += w[i] * pops[i]
            
            # Update path
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ y_w) / sigma
            sigma_path = np.linalg.norm(ps)
            
            # Update covariance
            delta_hs = (sigma_path > (1.4 + 2/(dim+1)) * chi_n)
            C = (1 + c_c * c_1 * delta_hs - c_1 - c_mu) * C
            C += c_1 * (ps * ps.T - C)
            for i in range(mu):
                y_i = (pops[i] - x) / sigma
                C += c_mu * w[i] * (y_i * y_i.T)
            
            # Ensure positive semi-definite
            C = (C + C.T) / 2
            try:
                L = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(L)
            except:
                # Regularize
                C += 1e-6 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                    C_inv_sqrt = np.linalg.inv(L)
                except:
                    L = np.eye(dim)
                    C_inv_sqrt = np.eye(dim)
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (sigma_path / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 100.0)
            
            # Update mean
            x = x + y_w
            x = np.clip(x, -5, 5)
            
            # Track best
            if fitvals[0] < best_f if 'best_f' in locals() else True:
                best_f = fitvals[0]
                best_x = pops[0].copy()
            else:
                if 'best_f' not in locals():
                    best_f = fitvals[0]
                    best_x = pops[0].copy()
            
            # Check for convergence
            if sigma < 1e-10:
                break

        # Initialize best if not set
        if 'best_f' not in locals():
            best_f = np.inf
            best_x = np.zeros(dim)
        
        # Phase 2: Local refinement around best point
        if remaining > 10 and time.time() < deadline - 5.0:
            x = best_x.copy()
            fx = best_f
            sigma_local = 0.1
            for _ in range(min(remaining, 50)):
                if remaining <= 0 or time.time() > deadline - 1.0:
                    break
                y = np.clip(x + sigma_local * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                remaining -= 1
                if fy < fx:
                    x, fx = y, fy
                    sigma_local *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma_local *= 0.8
                if sigma_local < 1e-4:
                    break

        return best_x, best_f


def sqrt(x):
    return np.sqrt(x)
# EVOLVE-BLOCK-END
